"""Numerical contract tests; these do not demonstrate model-quality gains."""

import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from replica_lab.neural import (
        FastState, FastWrite, LinearFastMemory, MemoryReader, PromptWorkspace,
        RecordEncoder,
    )


@unittest.skipIf(torch is None, "optional torch dependency is not installed")
class ReaderContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        self.hidden = torch.randn(2, 5, 16)
        self.support = torch.randn(2, 7, 16)

    def test_initial_output_is_exact_backbone_bypass(self):
        encoder = RecordEncoder(16, 8)
        reader = MemoryReader(16, 8)
        slots, mask = encoder(self.support)
        delta = reader(self.hidden, slots, mask)
        self.assertTrue(torch.equal(delta, torch.zeros_like(delta)))
        self.assertTrue(torch.equal(self.hidden + delta, self.hidden))

    def test_disabled_and_empty_context_stay_null_after_training(self):
        reader = MemoryReader(16, 8)
        torch.nn.init.normal_(reader.output_projection.weight)
        memory = torch.randn(2, 4, 8)
        self.assertTrue(torch.equal(reader(self.hidden, memory, enabled=False),
                                    torch.zeros_like(self.hidden)))
        for memory, mask in [(memory, torch.zeros(2, 4, dtype=torch.bool)),
                             (memory[:, :0], None)]:
            delta = reader(self.hidden, memory, mask)
            self.assertTrue(torch.isfinite(delta).all())
            self.assertTrue(torch.equal(delta, torch.zeros_like(delta)))

    def test_output_then_upstream_receive_gradients(self):
        encoder, reader = RecordEncoder(16, 8), MemoryReader(16, 8)
        optimizer = torch.optim.SGD(list(encoder.parameters()) + list(reader.parameters()),
                                    lr=0.2)
        slots, mask = encoder(self.support)
        loss = (self.hidden + reader(self.hidden, slots, mask)).square().mean()
        loss.backward()
        self.assertGreater(reader.output_projection.weight.grad.abs().sum().item(), 0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        slots, mask = encoder(self.support)
        (self.hidden + reader(self.hidden, slots, mask)).square().mean().backward()
        self.assertGreater(encoder.input_projection.weight.grad.abs().sum().item(), 0)
        self.assertGreater(reader.query_projection.weight.grad.abs().sum().item(), 0)

    def test_masked_record_content_is_inert_and_empty_rows_finite(self):
        encoder = RecordEncoder(16, 8)
        mask = torch.tensor([[1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]])
        changed = self.support.clone()
        changed[~mask.bool()] = 12345
        slots1, valid1 = encoder(self.support, mask)
        slots2, valid2 = encoder(changed, mask)
        self.assertTrue(torch.equal(slots1, slots2))
        self.assertTrue(torch.equal(valid1, valid2))
        self.assertTrue(torch.isfinite(slots1).all())
        self.assertTrue(torch.equal(slots1[1], torch.zeros_like(slots1[1])))
        zero, valid = encoder(self.support[:, :0])
        self.assertEqual(tuple(zero.shape), (2, 4, 8))
        self.assertFalse(valid.any())

    def test_reader_queries_do_not_mix_time_positions(self):
        reader = MemoryReader(16, 8)
        torch.nn.init.normal_(reader.output_projection.weight)
        memory = torch.randn(2, 4, 8)
        changed = self.hidden.clone()
        changed[:, 3:] += 100
        first, second = reader(self.hidden, memory), reader(changed, memory)
        self.assertTrue(torch.equal(first[:, :3], second[:, :3]))

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            RecordEncoder(16, 8)(self.support, torch.ones(2, 6))
        with self.assertRaises(ValueError):
            MemoryReader(16, 8)(self.hidden, torch.randn(1, 4, 8))


@unittest.skipIf(torch is None, "optional torch dependency is not installed")
class WorkspaceContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        torch.set_num_threads(1)

    def test_prompt_only_state_and_causal_queries(self):
        module = PromptWorkspace(16, 8)
        torch.nn.init.normal_(module.reader.output_projection.weight)
        prompt = torch.randn(2, 6, 16)
        state = module.build(prompt)
        target1 = torch.randn(2, 5, 16)
        target2 = target1.clone()
        target2[:, 2:] = torch.randn_like(target2[:, 2:])
        out1, out2 = module.read(target1, state), module.read(target2, state)
        self.assertTrue(torch.equal(out1[:, :2], out2[:, :2]))
        self.assertTrue(torch.equal(state, module.build(prompt)))
        # This test assumes target hidden states are themselves causally formed.

    def test_workspace_null_and_independent_gate(self):
        module = PromptWorkspace(16, 8)
        reader = MemoryReader(16, 8)
        self.assertIsNot(module.reader.gate_logit, reader.gate_logit)
        hidden = torch.randn(2, 5, 16)
        state = module.build(torch.randn(2, 6, 16), torch.zeros(2, 6))
        torch.nn.init.normal_(module.reader.output_projection.weight)
        delta = module.read(hidden, state)
        self.assertTrue(torch.isfinite(delta).all())
        self.assertTrue(torch.equal(delta, torch.zeros_like(delta)))
        nonempty = module.build(torch.randn(2, 6, 16))
        self.assertTrue(torch.equal(module.read(hidden, nonempty, enabled=False),
                                    torch.zeros_like(hidden)))

    def test_workspace_upstream_gradients_after_output_opens(self):
        module = PromptWorkspace(16, 8)
        optimizer = torch.optim.SGD(module.parameters(), lr=0.2)
        prompt, hidden = torch.randn(2, 6, 16), torch.randn(2, 5, 16)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            (hidden + module.read(hidden, module.build(prompt))).square().mean().backward()
            optimizer.step()
        self.assertGreater(module.recurrent.weight_hh.grad.abs().sum().item(), 0)
        self.assertGreater(module.encoder.input_projection.weight.grad.abs().sum().item(), 0)


@unittest.skipIf(torch is None, "optional torch dependency is not installed")
class FastMemoryContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(1)
        self.memory = LinearFastMemory(4, 3, learning_rate=0.2, max_update_norm=None)
        self.state = self.memory.initial_state(2)
        self.keys = torch.randn(2, 5, 4)
        self.values = torch.randn(2, 5, 3)

    def test_chunk_updates_are_future_only_and_do_not_mutate(self):
        before = self.state.weights.clone()
        old_read = self.memory.read(self.keys, self.state)
        new = self.memory.update_chunk(self.state, self.keys, self.values)
        self.assertTrue(torch.equal(before, self.state.weights))
        self.assertTrue(torch.equal(old_read, self.memory.read(self.keys, self.state)))
        self.assertFalse(torch.equal(old_read, self.memory.read(self.keys, new)))
        self.assertEqual(new.chunk_index, 1)
        self.assertEqual(self.state.chunk_index, 0)

    def test_shared_ports_aggregate_against_one_snapshot(self):
        write1 = FastWrite(self.keys[:, :2], self.values[:, :2], 0)
        write2 = FastWrite(self.keys[:, 2:], self.values[:, 2:], 0)
        combined = self.memory.update_shared_chunk(self.state, [write1, write2])
        single = self.memory.update_chunk(self.state, self.keys, self.values)
        self.assertTrue(torch.allclose(combined.weights, single.weights, atol=1e-7))
        self.assertEqual(combined.chunk_index, 1)
        with self.assertRaises(ValueError):
            self.memory.update_shared_chunk(combined, [write1])

    def test_sessions_and_empty_rows_are_isolated(self):
        changed = self.values.clone()
        changed[1] += 100
        first = self.memory.update_chunk(self.state, self.keys, self.values)
        second = self.memory.update_chunk(self.state, self.keys, changed)
        self.assertTrue(torch.equal(first.weights[0], second.weights[0]))
        self.assertFalse(torch.equal(first.weights[1], second.weights[1]))
        mask = torch.tensor([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]])
        sparse = self.memory.update_chunk(self.state, self.keys, self.values, mask)
        self.assertTrue(torch.equal(sparse.weights[1], self.state.weights[1]))
        self.assertTrue(torch.isfinite(sparse.weights).all())

    def test_differentiable_writes_reach_keys_values_and_prior_state(self):
        keys = self.keys.clone().requires_grad_()
        values = self.values.clone().requires_grad_()
        state = FastState(self.state.weights.clone().requires_grad_())
        updated = self.memory.update_chunk(state, keys, values)
        self.memory.read(torch.randn(2, 3, 4), updated).square().mean().backward()
        for tensor in (keys, values, state.weights):
            self.assertIsNotNone(tensor.grad)
            self.assertGreater(tensor.grad.abs().sum().item(), 0)

    def test_update_norm_is_bounded_per_session(self):
        memory = LinearFastMemory(4, 3, learning_rate=1, max_update_norm=0.1)
        result = memory.update_chunk(self.state, self.keys, self.values * 10000)
        norms = torch.linalg.vector_norm(result.weights - self.state.weights, dim=(1, 2))
        self.assertTrue((norms <= 0.100001).all())

    def test_all_empty_writes_preserve_weights_even_with_decay(self):
        memory = LinearFastMemory(4, 3, decay=0.2)
        state = FastState(torch.randn(2, 4, 3), 7)
        result = memory.update_chunk(state, self.keys, self.values, torch.zeros(2, 5))
        self.assertTrue(torch.equal(result.weights, state.weights))
        self.assertEqual(result.chunk_index, 8)


if __name__ == "__main__":
    unittest.main()
