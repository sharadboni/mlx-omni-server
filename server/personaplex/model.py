from __future__ import annotations

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from .config import (
    PersonaPlexConfig,
    SINE_TOKENS,
    SILENCE_TOKENS,
    DEFAULT_SYSTEM_PROMPT_TOKENS,
)
from .temporal import TemporalTransformer
from .depformer import Depformer
from .sampling import sample_top_k, sample_top_k_with_penalty
from .weight_loading import load_weights, load_voice


class PersonaPlexModel(nn.Module):
    """PersonaPlex speech-to-speech model (offline/batch mode)."""

    HF_REPO = "aufklarer/PersonaPlex-7B-MLX-4bit"

    def __init__(self, cfg: PersonaPlexConfig = PersonaPlexConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        self.temporal = TemporalTransformer(cfg.temporal)
        self.depformer = Depformer(cfg.depformer, temporal_dim=cfg.temporal.dim)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, repo_id: str = HF_REPO) -> "PersonaPlexModel":
        from huggingface_hub import snapshot_download
        import os
        local_dir = snapshot_download(repo_id=repo_id)
        model = cls()
        load_weights(model, local_dir)
        return model, local_dir

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def stream_respond(
        self,
        user_audio: np.ndarray,
        voice: str = "NATF2",
        system_prompt_tokens: list[int] | None = None,
        model_dir: str | None = None,
        max_steps: int = 500,
        mimi_file: str | None = None,
    ):
        """
        Generator version of respond(). Yields float32 PCM chunks (1920 samples each)
        as they are generated, enabling true streaming playback.
        """
        import rustymimi
        import os

        cfg = self.cfg
        nQ = cfg.temporal.nQ
        delays = cfg.delays
        max_delay = cfg.max_delay
        num_streams = cfg.num_streams
        text_padding_id = cfg.temporal.text_padding_id

        if mimi_file is None:
            if model_dir is None:
                raise ValueError("model_dir or mimi_file required")
            mimi_file = os.path.join(model_dir, "mimi.safetensors")

        audio_tokenizer = rustymimi.Tokenizer(mimi_file, num_codebooks=8)
        in_pcms = user_audio.reshape(1, -1).astype(np.float32)
        total_samples = in_pcms.shape[-1]
        steps_in = (total_samples + 1919) // 1920

        user_codes_list: list[np.ndarray] = []
        for idx in range(steps_in):
            start = idx * 1920
            end = min(start + 1920, total_samples)
            chunk = in_pcms[:, start:end]
            if chunk.shape[-1] < 1920:
                chunk = np.pad(chunk, ((0, 0), (0, 1920 - chunk.shape[-1])))
            encoded = audio_tokenizer.encode_step(chunk[None])
            enc = np.array(encoded).squeeze()
            if enc.ndim == 1:
                enc = enc.reshape(-1, 1)
            if enc.shape[0] == 1 and enc.shape[1] == 8:
                enc = enc.T
            user_codes_list.append(enc[:nQ, :])

        user_codes_np = np.concatenate(user_codes_list, axis=1)
        user_frame_count = user_codes_np.shape[1]

        voice_embeddings, voice_cache_arr = None, None
        if model_dir is not None:
            from .weight_loading import load_voice
            try:
                voice_embeddings, voice_cache_arr = load_voice(model_dir, voice)
            except FileNotFoundError:
                pass

        voice_frame_count = voice_embeddings.shape[0] if voice_embeddings is not None else 0
        silence_frames = int(0.5 * 12.5)
        text_prompt = system_prompt_tokens or DEFAULT_SYSTEM_PROMPT_TOKENS
        text_prompt_len = len(text_prompt)

        self.temporal.reset_cache()

        prompt_len = voice_frame_count + silence_frames + text_prompt_len + silence_frames
        prefill_len = prompt_len + user_frame_count
        CT = max_delay + 3
        total_len = prefill_len + max_steps + max_delay + 3

        token_cache = [[-1] * total_len for _ in range(num_streams)]

        def fill_phase(pos, nframes, text_tok, agent_toks, user_toks):
            for _ in range(nframes):
                token_cache[0][pos + delays[0]] = text_tok
                for cb in range(nQ):
                    token_cache[1 + cb][pos + delays[1 + cb]] = agent_toks[cb]
                    token_cache[1 + nQ + cb][pos + delays[1 + nQ + cb]] = user_toks[cb]
                pos += 1
            return pos

        pos = fill_phase(0, voice_frame_count, text_padding_id, SILENCE_TOKENS, SINE_TOKENS)

        if voice_cache_arr is not None and voice_frame_count > 0:
            vc = np.array(voice_cache_arr)
            for s in range(num_streams):
                d = delays[s]
                for k in range(d + 1):
                    flat_pos = voice_frame_count - 1 + k
                    ring_pos = (voice_frame_count + k) % CT
                    if 0 <= flat_pos < total_len:
                        token_cache[s][flat_pos] = int(vc[0, s, ring_pos])

        pos = fill_phase(pos, silence_frames, text_padding_id, SILENCE_TOKENS, SINE_TOKENS)

        for t in range(text_prompt_len):
            token_cache[0][pos + delays[0]] = text_prompt[t]
            for cb in range(nQ):
                token_cache[1 + cb][pos + delays[1 + cb]] = SILENCE_TOKENS[cb]
                token_cache[1 + nQ + cb][pos + delays[1 + nQ + cb]] = SINE_TOKENS[cb]
            pos += 1

        pos = fill_phase(pos, silence_frames, text_padding_id, SILENCE_TOKENS, SINE_TOKENS)

        for t in range(user_frame_count):
            token_cache[0][pos + delays[0]] = text_padding_id
            for cb in range(nQ):
                token_cache[1 + cb][pos + delays[1 + cb]] = SILENCE_TOKENS[cb]
                token_cache[1 + nQ + cb][pos + delays[1 + nQ + cb]] = int(user_codes_np[cb, t])
            pos += 1

        if voice_embeddings is not None and voice_frame_count > 0:
            import mlx.core as mx
            emb = voice_embeddings.reshape(voice_frame_count, self.cfg.temporal.dim)
            emb = mx.array(np.array(emb))[None]
            self.temporal.forward_batch_embedding(emb, offset=0)

        import mlx.core as mx
        non_voice_len = silence_frames + text_prompt_len + silence_frames
        if non_voice_len > 0:
            batch_text = []
            batch_audio = [[] for _ in range(num_streams - 1)]
            for t in range(non_voice_len):
                global_step = voice_frame_count + t
                read_idx = global_step - 1 if global_step > 0 else 0
                txt = token_cache[0][read_idx] if global_step > 0 else text_padding_id
                batch_text.append(txt)
                for stream in range(1, num_streams):
                    tok = token_cache[stream][read_idx] if global_step > 0 else -1
                    batch_audio[stream - 1].append(tok)
            text_arr = mx.array(batch_text, dtype=mx.int32)[None]
            audio_arr = mx.array(batch_audio, dtype=mx.int32)[None]
            hidden, _ = self.temporal.forward(text_arr, audio_arr, offset=voice_frame_count)
            mx.eval(hidden)

        # Initialize Mimi decoder once — state is maintained across frames for continuity
        decode_tokenizer = rustymimi.Tokenizer(mimi_file, num_codebooks=8)

        agent_tokens: list[list[int]] = [[] for _ in range(cfg.depformer.num_steps)]
        all_text_tokens: list[int] = []
        consecutive_silence = 0
        silence_early_stop = 8  # ~640ms of silence ends the response

        for step in range(prompt_len, prefill_len + max_steps):
            read_idx = step - 1
            text_tok = token_cache[0][read_idx]
            text_arr = mx.array([[text_tok]], dtype=mx.int32)
            audio_toks = [[token_cache[stream][read_idx]] for stream in range(1, num_streams)]
            audio_arr = mx.array(audio_toks, dtype=mx.int32)[None]

            hidden, text_logits = self.temporal.forward(text_arr, audio_arr, offset=step)

            text_history = all_text_tokens[-30:]
            text_token = sample_top_k_with_penalty(
                text_logits[:, 0, :], temperature=0.7, top_k=25,
                past_tokens=text_history, penalty=1.2,
            )
            mx.eval(text_token)
            text_val = int(text_token[0].item())

            provided_tokens = None
            if step < prefill_len:
                provided = [-1] * cfg.depformer.num_steps
                for cb in range(nQ):
                    user_stream = 1 + nQ + cb
                    if 0 <= step < total_len:
                        tok = token_cache[user_stream][step]
                        if tok >= 0:
                            provided[nQ + cb] = tok
                provided_tokens = provided

            def make_sample_fn(step_val):
                def sample_fn(logits, cb_idx):
                    history = agent_tokens[cb_idx][-30:]
                    return sample_top_k_with_penalty(
                        logits, temperature=0.8, top_k=250,
                        past_tokens=history, penalty=1.2,
                    )
                return sample_fn

            agent_codes = self.depformer.generate(
                temporal_hidden=hidden,
                text_token=text_token,
                provided_tokens=provided_tokens,
                sample_fn=make_sample_fn(step),
            )

            if step < total_len:
                token_cache[0][step] = text_val
            if step >= prefill_len:
                all_text_tokens.append(text_val)

            agent_arr = np.array(agent_codes[0])
            for cb in range(nQ):
                tok = int(agent_arr[cb])
                if step < total_len:
                    token_cache[1 + cb][step] = tok
                agent_tokens[cb].append(tok)

            for cb in range(nQ, cfg.depformer.num_steps):
                tok = int(agent_arr[cb])
                if step >= prefill_len and step < total_len:
                    token_cache[1 + cb][step] = tok
                agent_tokens[cb].append(tok)

            # Decode and yield this frame immediately; use audio energy for silence detection
            if step >= prefill_len:
                frame = np.array([[agent_tokens[cb][-1]] for cb in range(nQ)],
                                 dtype=np.uint32)  # [8, 1]
                frame = frame[None]  # [1, 8, 1]
                out = np.array(decode_tokenizer.decode_step(frame)).flatten().astype(np.float32)
                yield out

                if silence_early_stop > 0:
                    rms = float(np.sqrt(np.mean(out ** 2)))
                    if rms < 0.01:
                        consecutive_silence += 1
                    else:
                        consecutive_silence = 0
                    if consecutive_silence >= silence_early_stop:
                        break

    def respond(
        self,
        user_audio: np.ndarray,          # [num_samples] float32 at 24kHz
        voice: str = "NATF2",
        system_prompt_tokens: list[int] | None = None,
        model_dir: str | None = None,
        max_steps: int = 500,
        mimi_file: str | None = None,
    ) -> tuple[np.ndarray, list[int]]:
        """
        Returns (response_audio [num_samples], text_tokens).
        model_dir is the local HuggingFace cache directory for voice loading.
        """
        import rustymimi
        import os

        cfg = self.cfg
        nQ = cfg.temporal.nQ
        delays = cfg.delays
        max_delay = cfg.max_delay
        num_streams = cfg.num_streams
        text_padding_id = cfg.temporal.text_padding_id

        # Resolve mimi file
        if mimi_file is None:
            if model_dir is None:
                raise ValueError("model_dir or mimi_file required")
            mimi_file = os.path.join(model_dir, "mimi.safetensors")

        # 1. Encode user audio with Mimi (8 codebooks)
        audio_tokenizer = rustymimi.Tokenizer(mimi_file, num_codebooks=8)
        in_pcms = user_audio.reshape(1, -1).astype(np.float32)
        total_samples = in_pcms.shape[-1]
        steps_in = (total_samples + 1919) // 1920

        user_codes_list: list[np.ndarray] = []
        for idx in range(steps_in):
            start = idx * 1920
            end = min(start + 1920, total_samples)
            chunk = in_pcms[:, start:end]
            if chunk.shape[-1] < 1920:
                chunk = np.pad(chunk, ((0, 0), (0, 1920 - chunk.shape[-1])))
            encoded = audio_tokenizer.encode_step(chunk[None])  # [1, 1, 8, 1]
            # shape is [batch, channels, codebooks, time] — extract [8, 1]
            enc = np.array(encoded)
            # Handle shape variations
            enc = enc.squeeze()
            if enc.ndim == 1:
                enc = enc.reshape(-1, 1)
            elif enc.ndim == 2:
                pass  # [8, 1] or [1, 8]
            # Ensure [codebooks, 1]
            if enc.shape[0] == 1 and enc.shape[1] == 8:
                enc = enc.T
            user_codes_list.append(enc[:nQ, :])  # [8, 1]

        user_codes_np = np.concatenate(user_codes_list, axis=1)  # [8, T]
        user_frame_count = user_codes_np.shape[1]

        # 2. Load voice embeddings + cache
        voice_embeddings, voice_cache_arr = None, None
        if model_dir is not None:
            try:
                voice_embeddings, voice_cache_arr = load_voice(model_dir, voice)
            except FileNotFoundError:
                pass

        voice_frame_count = voice_embeddings.shape[0] if voice_embeddings is not None else 0
        silence_frames = int(0.5 * 12.5)  # 6 frames @ 12.5 Hz
        text_prompt = system_prompt_tokens or DEFAULT_SYSTEM_PROMPT_TOKENS
        text_prompt_len = len(text_prompt)

        # 3. Reset caches
        self.temporal.reset_cache()

        prompt_len = voice_frame_count + silence_frames + text_prompt_len + silence_frames
        prefill_len = prompt_len + user_frame_count
        CT = max_delay + 3  # ring buffer size = 4
        total_len = prefill_len + max_steps + max_delay + 3

        # 4. Initialize token cache [17, total_len] with -1
        token_cache = [[-1] * total_len for _ in range(num_streams)]

        def fill_phase(pos: int, nframes: int, text_tok: int,
                       agent_toks: list[int], user_toks: list[int]) -> int:
            for _ in range(nframes):
                token_cache[0][pos + delays[0]] = text_tok
                for cb in range(nQ):
                    token_cache[1 + cb][pos + delays[1 + cb]] = agent_toks[cb]
                    token_cache[1 + nQ + cb][pos + delays[1 + nQ + cb]] = user_toks[cb]
                pos += 1
            return pos

        # Phase 1: Voice prompt
        pos = fill_phase(0, voice_frame_count, text_padding_id, SILENCE_TOKENS, SINE_TOKENS)

        # Apply voice cache ring buffer to last few positions
        if voice_cache_arr is not None and voice_frame_count > 0:
            vc = np.array(voice_cache_arr)  # [1, 17, CT]
            for s in range(num_streams):
                d = delays[s]
                for k in range(d + 1):
                    flat_pos = voice_frame_count - 1 + k
                    ring_pos = (voice_frame_count + k) % CT
                    if 0 <= flat_pos < total_len:
                        token_cache[s][flat_pos] = int(vc[0, s, ring_pos])

        # Phase 2: Silence 1
        pos = fill_phase(pos, silence_frames, text_padding_id, SILENCE_TOKENS, SINE_TOKENS)

        # Phase 3: Text prompt (stream 0 gets actual tokens)
        for t in range(text_prompt_len):
            token_cache[0][pos + delays[0]] = text_prompt[t]
            for cb in range(nQ):
                token_cache[1 + cb][pos + delays[1 + cb]] = SILENCE_TOKENS[cb]
                token_cache[1 + nQ + cb][pos + delays[1 + nQ + cb]] = SINE_TOKENS[cb]
            pos += 1

        # Phase 4: Silence 2
        pos = fill_phase(pos, silence_frames, text_padding_id, SILENCE_TOKENS, SINE_TOKENS)

        # Phase 5: User audio
        for t in range(user_frame_count):
            token_cache[0][pos + delays[0]] = text_padding_id
            for cb in range(nQ):
                token_cache[1 + cb][pos + delays[1 + cb]] = SILENCE_TOKENS[cb]
                token_cache[1 + nQ + cb][pos + delays[1 + nQ + cb]] = int(user_codes_np[cb, t])
            pos += 1

        # 5. Batched prefill

        # Voice prompt: feed pre-computed embeddings
        if voice_embeddings is not None and voice_frame_count > 0:
            # [T, 1, 1, dim] → [1, T, dim]
            emb = voice_embeddings.reshape(voice_frame_count, self.cfg.temporal.dim)
            emb = mx.array(np.array(emb))[None]  # [1, T, dim]
            self.temporal.forward_batch_embedding(emb, offset=0)

        # Silence + text prompt + silence: batched token forward
        non_voice_len = silence_frames + text_prompt_len + silence_frames
        if non_voice_len > 0:
            batch_text = []
            batch_audio = [[] for _ in range(num_streams - 1)]
            for t in range(non_voice_len):
                global_step = voice_frame_count + t
                read_idx = global_step - 1 if global_step > 0 else 0
                txt = token_cache[0][read_idx] if global_step > 0 else text_padding_id
                batch_text.append(txt)
                for stream in range(1, num_streams):
                    tok = token_cache[stream][read_idx] if global_step > 0 else -1
                    batch_audio[stream - 1].append(tok)

            text_arr = mx.array(batch_text, dtype=mx.int32)[None]  # [1, T]
            audio_arr = mx.array(batch_audio, dtype=mx.int32)[None]  # [1, 16, T]
            hidden, _ = self.temporal.forward(text_arr, audio_arr, offset=voice_frame_count)
            mx.eval(hidden)

        # 6. Per-step generation
        agent_tokens: list[list[int]] = [[] for _ in range(cfg.depformer.num_steps)]
        all_text_tokens: list[int] = []
        consecutive_silence = 0
        silence_early_stop = 8  # ~640ms of silence ends the response

        for step in range(prompt_len, prefill_len + max_steps):
            read_idx = step - 1
            text_tok = token_cache[0][read_idx]
            text_arr = mx.array([[text_tok]], dtype=mx.int32)  # [1, 1]
            audio_toks = [[token_cache[stream][read_idx]] for stream in range(1, num_streams)]
            audio_arr = mx.array(audio_toks, dtype=mx.int32)[None]  # [1, 16, 1]

            hidden, text_logits = self.temporal.forward(text_arr, audio_arr, offset=step)

            # Sample text token
            text_history = all_text_tokens[-30:]
            text_token = sample_top_k_with_penalty(
                text_logits[:, 0, :], temperature=0.7, top_k=25,
                past_tokens=text_history, penalty=1.2,
            )
            mx.eval(text_token)
            text_val = int(text_token[0].item())

            # Build provided tokens for depformer during user audio phase
            provided_tokens = None
            if step < prefill_len:
                provided = [-1] * cfg.depformer.num_steps
                for cb in range(nQ):
                    user_stream = 1 + nQ + cb
                    if 0 <= step < total_len:
                        tok = token_cache[user_stream][step]
                        if tok >= 0:
                            provided[nQ + cb] = tok
                provided_tokens = provided

            # Generate audio codes via depformer
            def make_sample_fn(step_val):
                def sample_fn(logits: mx.array, cb_idx: int) -> mx.array:
                    history = agent_tokens[cb_idx][-30:]
                    return sample_top_k_with_penalty(
                        logits, temperature=0.8, top_k=250,
                        past_tokens=history, penalty=1.2,
                    )
                return sample_fn

            agent_codes = self.depformer.generate(
                temporal_hidden=hidden,
                text_token=text_token,
                provided_tokens=provided_tokens,
                sample_fn=make_sample_fn(step),
            )

            # Write to token cache
            if step < total_len:
                token_cache[0][step] = text_val
            if step >= prefill_len:
                all_text_tokens.append(text_val)

            agent_arr = np.array(agent_codes[0])  # [num_steps]
            for cb in range(nQ):
                tok = int(agent_arr[cb])
                if step < total_len:
                    token_cache[1 + cb][step] = tok
                agent_tokens[cb].append(tok)

            for cb in range(nQ, cfg.depformer.num_steps):
                tok = int(agent_arr[cb])
                if step >= prefill_len and step < total_len:
                    token_cache[1 + cb][step] = tok
                agent_tokens[cb].append(tok)

            # Silence early stopping
            if step >= prefill_len and silence_early_stop > 0:
                is_silence = all(
                    agent_tokens[cb][-1] == SILENCE_TOKENS[cb] for cb in range(nQ)
                )
                consecutive_silence = consecutive_silence + 1 if is_silence else 0
                if consecutive_silence >= silence_early_stop:
                    break

        if not agent_tokens[0]:
            return np.zeros(0, dtype=np.float32), all_text_tokens

        # 7. Decode agent audio tokens with Mimi (8 codebooks)
        num_frames = len(agent_tokens[0])
        decode_tokenizer = rustymimi.Tokenizer(mimi_file, num_codebooks=8)
        out_pcm_list: list[np.ndarray] = []

        for t in range(num_frames):
            frame = np.array([[agent_tokens[cb][t]] for cb in range(nQ)],
                             dtype=np.uint32)  # [8, 1]
            frame = frame[None]  # [1, 8, 1]  — rustymimi expects [batch, codebooks, time], uint32
            out = decode_tokenizer.decode_step(frame)
            out_pcm_list.append(np.array(out))  # [1, 1, 1920]

        if not out_pcm_list:
            return np.zeros(0, dtype=np.float32), all_text_tokens

        out_pcm = np.concatenate(out_pcm_list, axis=-1)  # [1, 1, num_frames*1920]
        audio_out = out_pcm.flatten()
        return audio_out.astype(np.float32), all_text_tokens

    def create_stream_session(
        self,
        voice: str = "NATF2",
        model_dir: str | None = None,
        system_prompt_tokens: list[int] | None = None,
        mimi_file: str | None = None,
    ) -> "PersonaPlexStreamSession":
        """Create a stateful real-time streaming session."""
        return PersonaPlexStreamSession(
            self, voice=voice, model_dir=model_dir,
            system_prompt_tokens=system_prompt_tokens, mimi_file=mimi_file,
        )


class PersonaPlexStreamSession:
    """
    Stateful real-time S2S session.

    Usage:
        session = model.create_stream_session(voice="NATF2", model_dir=...)
        # Then for each 1920-sample PCM frame from the microphone:
        agent_pcm = session.process_frame(user_pcm)  # returns 1920 samples
    """

    FRAME_SIZE = 1920  # samples per frame at 12.5 Hz / 24kHz

    def __init__(
        self,
        model: "PersonaPlexModel",
        voice: str = "NATF2",
        model_dir: str | None = None,
        system_prompt_tokens: list[int] | None = None,
        mimi_file: str | None = None,
    ) -> None:
        import rustymimi
        import os

        self._model = model
        cfg = model.cfg
        self._cfg = cfg
        self._nQ = cfg.temporal.nQ
        self._delays = cfg.delays
        self._num_streams = cfg.num_streams
        self._text_padding_id = cfg.temporal.text_padding_id

        if mimi_file is None:
            if model_dir is None:
                raise ValueError("model_dir or mimi_file required")
            mimi_file = os.path.join(model_dir, "mimi.safetensors")

        # Separate encoder/decoder instances to maintain independent Mimi state
        self._encoder = rustymimi.Tokenizer(mimi_file, num_codebooks=8)
        self._decoder = rustymimi.Tokenizer(mimi_file, num_codebooks=8)

        voice_embeddings, voice_cache_arr = None, None
        if model_dir is not None:
            try:
                voice_embeddings, voice_cache_arr = load_voice(model_dir, voice)
            except FileNotFoundError:
                pass

        text_prompt = system_prompt_tokens or DEFAULT_SYSTEM_PROMPT_TOKENS
        nQ = self._nQ
        delays = self._delays
        num_streams = self._num_streams
        voice_frame_count = voice_embeddings.shape[0] if voice_embeddings is not None else 0
        silence_frames = int(0.5 * 12.5)  # 6 frames
        max_delay = cfg.max_delay
        CT = max_delay + 3

        # Dynamically-growing token cache (lists, indexed by step)
        self._cache: list[list[int]] = [[] for _ in range(num_streams)]

        def _ensure(pos: int) -> None:
            for s in range(num_streams):
                while len(self._cache[s]) <= pos:
                    self._cache[s].append(-1)

        def fill_phase(pos: int, nframes: int, text_tok: int,
                       agent_toks: list[int], user_toks: list[int]) -> int:
            for _ in range(nframes):
                _ensure(pos + max_delay + 1)
                self._cache[0][pos + delays[0]] = text_tok
                for cb in range(nQ):
                    self._cache[1 + cb][pos + delays[1 + cb]] = agent_toks[cb]
                    self._cache[1 + nQ + cb][pos + delays[1 + nQ + cb]] = user_toks[cb]
                pos += 1
            return pos

        # ---- Prefill ----
        model.temporal.reset_cache()

        pos = fill_phase(0, voice_frame_count, self._text_padding_id, SILENCE_TOKENS, SINE_TOKENS)

        if voice_cache_arr is not None and voice_frame_count > 0:
            vc = np.array(voice_cache_arr)
            for s in range(num_streams):
                d = delays[s]
                for k in range(d + 1):
                    flat_pos = voice_frame_count - 1 + k
                    ring_pos = (voice_frame_count + k) % CT
                    _ensure(flat_pos + 1)
                    self._cache[s][flat_pos] = int(vc[0, s, ring_pos])

        pos = fill_phase(pos, silence_frames, self._text_padding_id, SILENCE_TOKENS, SINE_TOKENS)

        for t in range(len(text_prompt)):
            _ensure(pos + max_delay + 1)
            self._cache[0][pos + delays[0]] = text_prompt[t]
            for cb in range(nQ):
                self._cache[1 + cb][pos + delays[1 + cb]] = SILENCE_TOKENS[cb]
                self._cache[1 + nQ + cb][pos + delays[1 + nQ + cb]] = SINE_TOKENS[cb]
            pos += 1

        pos = fill_phase(pos, silence_frames, self._text_padding_id, SILENCE_TOKENS, SINE_TOKENS)

        # Run batched prefill through the temporal transformer
        if voice_embeddings is not None and voice_frame_count > 0:
            emb = voice_embeddings.reshape(voice_frame_count, cfg.temporal.dim)
            emb = mx.array(np.array(emb))[None]
            model.temporal.forward_batch_embedding(emb, offset=0)

        non_voice_len = silence_frames + len(text_prompt) + silence_frames
        if non_voice_len > 0:
            batch_text, batch_audio = [], [[] for _ in range(num_streams - 1)]
            for t in range(non_voice_len):
                gs = voice_frame_count + t
                ri = gs - 1 if gs > 0 else 0
                batch_text.append(self._cache[0][ri] if gs > 0 else self._text_padding_id)
                for stream in range(1, num_streams):
                    batch_audio[stream - 1].append(self._cache[stream][ri] if gs > 0 else -1)
            hidden, _ = model.temporal.forward(
                mx.array(batch_text, dtype=mx.int32)[None],
                mx.array(batch_audio, dtype=mx.int32)[None],
                offset=voice_frame_count,
            )
            mx.eval(hidden)

        self._step = pos  # first streaming step
        self._agent_history: list[list[int]] = [[] for _ in range(cfg.depformer.num_steps)]
        self._text_history: list[int] = []

    # ------------------------------------------------------------------

    def process_frame(self, user_pcm: np.ndarray) -> np.ndarray:
        """
        Process one frame of user audio (1920 samples at 24kHz, float32).
        Returns one frame of agent audio (1920 samples, float32).
        Called in real time at ~12.5 Hz.
        """
        nQ = self._nQ
        cfg = self._cfg
        step = self._step

        # Encode user audio frame → 8 codebook tokens
        chunk = user_pcm.reshape(1, -1).astype(np.float32)
        if chunk.shape[-1] < self.FRAME_SIZE:
            chunk = np.pad(chunk, ((0, 0), (0, self.FRAME_SIZE - chunk.shape[-1])))
        enc = np.array(self._encoder.encode_step(chunk[None])).squeeze()
        if enc.ndim == 1:
            enc = enc.reshape(-1, 1)
        if enc.shape[0] == 1 and enc.shape[1] == 8:
            enc = enc.T
        user_tokens = enc[:nQ, 0].astype(int)  # [8]

        # Extend cache for this step and the next (delayed writes)
        while len(self._cache[0]) <= step + cfg.max_delay + 1:
            for s in range(self._num_streams):
                self._cache[s].append(-1)

        # Write user tokens into cache (read by next step's temporal transformer)
        for cb in range(nQ):
            self._cache[1 + nQ + cb][step] = int(user_tokens[cb])

        # Read previous step's tokens for temporal transformer input
        read_idx = step - 1
        text_tok = self._cache[0][read_idx]
        audio_toks = [[self._cache[stream][read_idx]] for stream in range(1, self._num_streams)]

        hidden, text_logits = self._model.temporal.forward(
            mx.array([[text_tok]], dtype=mx.int32),
            mx.array(audio_toks, dtype=mx.int32)[None],
            offset=step,
        )

        # Sample text token
        text_token = sample_top_k_with_penalty(
            text_logits[:, 0, :], temperature=0.7, top_k=25,
            past_tokens=self._text_history[-30:], penalty=1.2,
        )
        mx.eval(text_token)
        text_val = int(text_token[0].item())
        self._cache[0][step] = text_val
        self._text_history.append(text_val)

        # Depformer — always provide current user tokens as conditioning
        provided = [-1] * cfg.depformer.num_steps
        for cb in range(nQ):
            provided[nQ + cb] = int(user_tokens[cb])

        def _sample_fn(logits: mx.array, cb_idx: int) -> mx.array:
            return sample_top_k_with_penalty(
                logits, temperature=0.8, top_k=250,
                past_tokens=self._agent_history[cb_idx][-30:], penalty=1.2,
            )

        agent_codes = self._model.depformer.generate(
            temporal_hidden=hidden,
            text_token=text_token,
            provided_tokens=provided,
            sample_fn=_sample_fn,
        )

        # Write agent tokens to cache and history
        agent_arr = np.array(agent_codes[0])
        for cb in range(nQ):
            tok = int(agent_arr[cb])
            self._cache[1 + cb][step] = tok
            self._agent_history[cb].append(tok)
        for cb in range(nQ, cfg.depformer.num_steps):
            self._agent_history[cb].append(int(agent_arr[cb]))

        # Decode agent tokens → PCM
        frame = np.array([[self._agent_history[cb][-1]] for cb in range(nQ)],
                         dtype=np.uint32)[None]  # [1, 8, 1]
        out = self._decoder.decode_step(frame)

        self._step += 1
        return np.array(out).flatten().astype(np.float32)
