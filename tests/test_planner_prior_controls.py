import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from mpail2.planner import Planner


class PlannerPriorControlsTest(unittest.TestCase):
    def test_step_seeds_sampler_with_shifted_previous_controls(self):
        fake = SimpleNamespace()
        fake.u_per_command = 1
        fake.cfg = SimpleNamespace(opt_iters=2)
        fake._opt_controls = torch.tensor([[[0.1], [0.2], [0.3]]])
        fake.sampling = SimpleNamespace(reset_iter_state=Mock())
        fake.update = Mock()
        fake.optimize = Mock(return_value=fake._opt_controls)

        Planner.step(fake, obs={"policy": torch.zeros(1, 3)}, use_prev_opt=True)

        expected = torch.tensor([[[0.2], [0.3], [0.0]]])
        torch.testing.assert_close(fake._opt_controls, expected)
        fake.sampling.reset_iter_state.assert_called_once()
        seeded_controls = fake.sampling.reset_iter_state.call_args.kwargs["prev_controls"]
        torch.testing.assert_close(seeded_controls, expected)


if __name__ == "__main__":
    unittest.main()
