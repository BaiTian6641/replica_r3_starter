"""Numerical E1/E2 contracts; these tests make no model-quality claim."""

import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from replica_lab.architecture.ports import MemoryAccessPort, SensePort
    from replica_lab.memory_system.writer import (
        ContextualRecordWriter, TrainableRecordReranker,
    )


@unittest.skipIf(torch is None, "optional torch dependency is not installed")
class MemoryAccessContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(59)
        torch.set_num_threads(1)
        self.hidden = torch.randn(2, 5, 16)
        self.memory = torch.randn(2, 7, 8)
        self.valid = torch.ones(2, 7, dtype=torch.bool)

    def test_projection_inventory_and_initial_exact_bypass(self):
        port = MemoryAccessPort(16, 8, 2)
        projections = [port.query_projection, port.key_projection,
                       port.value_projection, port.output_projection]
        self.assertEqual(sum(p.weight.numel() for p in projections), 2 * 16 * 8 + 2 * 8 * 8)
        for projection in projections[:-1]:
            self.assertGreater(projection.weight.abs().sum().item(), 0)
        delta = port(self.hidden, self.memory, self.valid)
        self.assertTrue(torch.equal(delta, torch.zeros_like(self.hidden)))
        self.assertTrue(torch.equal(self.hidden + delta, self.hidden))

    def test_output_opens_before_upstream_factors_and_source_writer(self):
        writer = ContextualRecordWriter(16, 8, 2, 2)
        port = MemoryAccessPort(16, 8, 2)
        source = torch.randn(2, 3, 6, 16, requires_grad=True)
        source_mask = torch.ones(2, 3, 6, dtype=torch.bool)
        optimizer = torch.optim.SGD([*writer.parameters(), *port.parameters()], lr=0.2)
        slots, mask = writer(source, source_mask)
        delta = port(self.hidden, slots.flatten(1, 2), mask.flatten(1, 2))
        (self.hidden + delta).square().mean().backward()
        self.assertGreater(port.output_projection.weight.grad.abs().sum().item(), 0)
        self.assertEqual(port.query_projection.weight.grad.abs().sum().item(), 0)
        self.assertEqual(source.grad.abs().sum().item(), 0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        source.grad = None
        slots, mask = writer(source, source_mask)
        delta = port(self.hidden, slots.flatten(1, 2), mask.flatten(1, 2))
        (self.hidden + delta).square().mean().backward()
        for parameter in (port.query_projection.weight, port.key_projection.weight,
                          port.value_projection.weight,
                          writer.encoder.input_projection.weight, source):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_empty_disabled_and_all_masked_stay_null_with_nonzero_parameters(self):
        port = MemoryAccessPort(16, 8, 2)
        torch.nn.init.normal_(port.output_projection.weight)
        for memory, valid, enabled in (
            (self.memory, self.valid, False),
            (self.memory[:, :0], self.valid[:, :0], True),
            (self.memory, torch.zeros_like(self.valid), True),
        ):
            delta = port(self.hidden, memory, valid, enabled)
            self.assertTrue(torch.isfinite(delta).all())
            self.assertTrue(torch.equal(delta, torch.zeros_like(self.hidden)))
        self.assertEqual(port(self.hidden[:, :0], self.memory, self.valid).shape, (2, 0, 16))
    def test_attention_audit_uses_head_layout_and_masks_empty_rows(self):
        port = MemoryAccessPort(16, 8, 2)
        torch.nn.init.normal_(port.output_projection.weight)
        valid = self.valid.clone()
        valid[1] = False
        delta, attention = port(self.hidden, self.memory, valid,
                                return_attention=True)
        self.assertEqual(attention["query"].shape, attention["read"].shape)
        self.assertTrue(torch.allclose(attention["weights"] @ attention["value"],
                                       attention["read"], atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.equal(attention["weights"][1],
                                    torch.zeros_like(attention["weights"][1])))
        self.assertTrue(torch.equal(delta[1], torch.zeros_like(delta[1])))

    def test_padding_nan_is_inert_and_queries_and_batch_rows_are_independent(self):
        port = MemoryAccessPort(16, 8, 2)
        torch.nn.init.normal_(port.output_projection.weight)
        valid = self.valid.clone()
        valid[0, 4:] = False
        valid[1] = False
        first = port(self.hidden, self.memory, valid)
        changed_memory = self.memory.clone()
        changed_memory[~valid] = float("nan")
        changed_hidden = self.hidden.clone()
        changed_hidden[0, 3:] *= -3
        second = port(changed_hidden, changed_memory, valid)
        self.assertTrue(torch.equal(first[0, :3], second[0, :3]))
        self.assertTrue(torch.equal(second[1], torch.zeros_like(second[1])))
        self.assertTrue(torch.isfinite(second).all())
        separate = port(self.hidden[:1], self.memory[:1], valid[:1])
        self.assertTrue(torch.allclose(first[:1], separate, atol=1e-7))

    def test_memory_permutation_and_separate_port_parameter_ownership(self):
        first, second = MemoryAccessPort(16, 8, 2), MemoryAccessPort(16, 8, 2)
        torch.nn.init.normal_(first.output_projection.weight)
        permutation = torch.tensor([3, 0, 6, 1, 4, 5, 2])
        expected = first(self.hidden, self.memory, self.valid)
        actual = first(self.hidden, self.memory[:, permutation], self.valid[:, permutation])
        self.assertTrue(torch.allclose(expected, actual, atol=2e-6))
        for p, q in zip(first.parameters(), second.parameters()):
            self.assertNotEqual(p.data_ptr(), q.data_ptr())

    def test_shape_errors_are_explicit(self):
        port = MemoryAccessPort(16, 8, 2)
        with self.assertRaises(ValueError):
            port(self.hidden, self.memory[:1], self.valid[:1])
        with self.assertRaises(ValueError):
            port(self.hidden, self.memory, self.valid[:, :6])
        with self.assertRaises(ValueError):
            MemoryAccessPort(16, 9, 2)


@unittest.skipIf(torch is None, "optional torch dependency is not installed")
class ContextualWriterContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(61)
        torch.set_num_threads(1)
        self.writer = ContextualRecordWriter(16, 8, 4, 2)
        self.hidden = torch.randn(2, 3, 6, 16)
        self.valid = torch.ones(2, 3, 6, dtype=torch.bool)

    def test_record_axes_survive_and_content_does_not_cross_records(self):
        expected, mask = self.writer(self.hidden, self.valid)
        self.assertEqual(expected.shape, (2, 3, 4, 8))
        self.assertEqual(mask.shape, (2, 3, 4))
        changed = self.hidden.clone()
        changed[1, 2] = torch.randn_like(changed[1, 2]) * 7
        actual, _ = self.writer(changed, self.valid)
        self.assertTrue(torch.equal(expected[0], actual[0]))
        self.assertTrue(torch.equal(expected[1, :2], actual[1, :2]))
        self.assertFalse(torch.equal(expected[1, 2], actual[1, 2]))
        isolated, _ = self.writer(self.hidden[1:2, 2:3], self.valid[1:2, 2:3])
        self.assertTrue(torch.allclose(expected[1:2, 2:3], isolated, atol=1e-6))

    def test_record_permutation_equivariance(self):
        expected, mask = self.writer(self.hidden, self.valid)
        permutation = torch.tensor([2, 0, 1])
        actual, actual_mask = self.writer(self.hidden[:, permutation], self.valid[:, permutation])
        self.assertTrue(torch.equal(expected[:, permutation], actual))
        self.assertTrue(torch.equal(mask[:, permutation], actual_mask))

    def test_masked_nan_values_do_not_affect_records_or_gradients(self):
        valid = self.valid.clone()
        valid[0, 1, 3:] = False
        valid[1, 2] = False
        expected, mask = self.writer(self.hidden, valid)
        changed = self.hidden.clone()
        changed[~valid] = float("nan")
        changed.requires_grad_()
        actual, actual_mask = self.writer(changed, valid)
        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.equal(mask, actual_mask))
        self.assertTrue(torch.isfinite(actual).all())
        self.assertTrue(torch.equal(actual[1, 2], torch.zeros_like(actual[1, 2])))
        self.assertFalse(actual_mask[1, 2].any())
        (actual * torch.randn_like(actual)).sum().backward()
        self.assertTrue(torch.isfinite(changed.grad).all())
        self.assertEqual(changed.grad[~valid].abs().sum().item(), 0)
        self.assertGreater(changed.grad[valid].abs().sum().item(), 0)

    def test_literal_empty_axes_have_explicit_output_shapes(self):
        for hidden, valid, shape in (
            (self.hidden[:, :0], self.valid[:, :0], (2, 0, 4, 8)),
            (self.hidden[:, :, :0], self.valid[:, :, :0], (2, 3, 4, 8)),
            (self.hidden[:0], self.valid[:0], (0, 3, 4, 8)),
        ):
            slots, mask = self.writer(hidden, valid)
            self.assertEqual(tuple(slots.shape), shape)
            self.assertEqual(tuple(mask.shape), shape[:3])
            self.assertEqual(slots.count_nonzero().item(), 0)
            self.assertFalse(mask.any())

    def test_shape_errors_are_explicit(self):
        with self.assertRaises(ValueError):
            self.writer(self.hidden.flatten(0, 1), self.valid.flatten(0, 1))
        with self.assertRaises(ValueError):
            self.writer(self.hidden, self.valid[:, :, :5])


@unittest.skipIf(torch is None, "optional torch dependency is not installed")
class SenseAndRankingContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(67)
        torch.set_num_threads(1)

    def test_sense_masked_padding_empty_rows_bound_and_gradient(self):
        sense = SensePort(16, 8)
        hidden = torch.randn(2, 5, 16, requires_grad=True)
        valid = torch.tensor([[True, True, False, False, False], [False] * 5])
        summary, mask = sense(hidden, valid)
        changed = hidden.detach().clone()
        changed[~valid] = float("nan")
        actual, actual_mask = sense(changed, valid)
        self.assertEqual(summary.shape, (2, 1, 8))
        self.assertTrue(torch.equal(summary, actual))
        self.assertTrue(torch.equal(mask, actual_mask))
        self.assertLessEqual(summary.abs().max().item(), 1)
        self.assertTrue(torch.equal(summary[1], torch.zeros_like(summary[1])))
        summary.square().sum().backward()
        self.assertGreater(hidden.grad[valid].abs().sum().item(), 0)
        self.assertEqual(hidden.grad[~valid].abs().sum().item(), 0)
        empty, mask = sense(hidden[:, :0], valid[:, :0])
        self.assertEqual(empty.shape, (2, 1, 8))
        self.assertFalse(mask.any())

    def test_reranker_masks_permutation_batching_and_upstream_gradients(self):
        ranker = TrainableRecordReranker(8)
        query = torch.randn(2, 8, requires_grad=True)
        candidates = torch.randn(2, 4, 8, requires_grad=True)
        valid = torch.tensor([[True, False, True, True], [False] * 4])
        expected = ranker(query, candidates, valid)
        self.assertTrue(torch.isfinite(expected[valid]).all())
        self.assertTrue(torch.isneginf(expected[~valid]).all())
        changed = candidates.detach().clone()
        changed[~valid] = float("nan")
        self.assertTrue(torch.equal(expected, ranker(query, changed, valid)))
        permutation = torch.tensor([2, 3, 1, 0])
        actual = ranker(query, candidates[:, permutation], valid[:, permutation])
        self.assertTrue(torch.equal(expected[:, permutation], actual))
        isolated = ranker(query[:1], candidates[:1], valid[:1])
        self.assertTrue(torch.allclose(expected[:1], isolated, atol=1e-7))
        expected[valid].square().sum().backward()
        self.assertGreater(query.grad[0].abs().sum().item(), 0)
        self.assertGreater(candidates.grad[valid].abs().sum().item(), 0)
        self.assertEqual(candidates.grad[~valid].abs().sum().item(), 0)
        self.assertGreater(ranker.query_projection.weight.grad.abs().sum().item(), 0)
        self.assertGreater(ranker.record_projection.weight.grad.abs().sum().item(), 0)
        self.assertEqual(ranker(query, candidates[:, :0], valid[:, :0]).shape, (2, 0))

    def test_sense_and_reranker_shape_errors_are_explicit(self):
        with self.assertRaises(ValueError):
            SensePort(16, 8)(torch.randn(2, 4, 16), torch.ones(2, 5))
        with self.assertRaises(ValueError):
            TrainableRecordReranker(8)(torch.randn(2, 8), torch.randn(1, 4, 8), torch.ones(1, 4))
        with self.assertRaises(ValueError):
            TrainableRecordReranker(8)(torch.randn(2, 8), torch.randn(2, 4, 8), torch.ones(2, 3))
