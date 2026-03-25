import pytest
import torch

from robomme.robomme_env.utils import swap_contact_monitoring as swapContact


pytestmark = [pytest.mark.lightweight]


class _Actor:
    def __init__(self, name: str):
        self.name = name


class _Scene:
    def __init__(self, forces):
        self.forces = forces
        self.calls = []

    def get_pairwise_contact_forces(self, actor_a, actor_b):
        key = tuple(sorted((actor_a.name, actor_b.name)))
        self.calls.append(key)
        force = self.forces.get(key, (0.0, 0.0, 0.0))
        return torch.tensor([force], dtype=torch.float32)


def _build_inputs(scene):
    actors = [_Actor("bin_0"), _Actor("bin_1"), _Actor("bin_2")]
    swap_schedule = [(None, None, 10, 20)]
    log_context = {
        "env": "ButtonUnmaskSwap",
        "episode": 7,
        "seed": 13,
    }
    state = swapContact.new_swap_contact_state()
    return actors, swap_schedule, log_context, state


def test_detect_swap_contacts_only_during_swap_window():
    scene = _Scene({("bin_0", "bin_2"): (0.0, 0.0, 2.5)})
    actors, swap_schedule, log_context, state = _build_inputs(scene)

    swapContact.detect_swap_contacts(
        scene=scene,
        actors=actors,
        swap_schedule=swap_schedule,
        timestep=9,
        state=state,
        log_context=log_context,
    )
    assert scene.calls == []
    assert swapContact.get_swap_contact_summary(state)["swap_contact_detected"] is False

    swapContact.detect_swap_contacts(
        scene=scene,
        actors=actors,
        swap_schedule=swap_schedule,
        timestep=10,
        state=state,
        log_context=log_context,
    )

    assert scene.calls == [
        ("bin_0", "bin_1"),
        ("bin_0", "bin_2"),
        ("bin_1", "bin_2"),
    ]
    summary = swapContact.get_swap_contact_summary(state)
    assert summary["swap_contact_detected"] is True
    assert summary["first_contact_step"] == 10
    assert summary["contact_pairs"] == ["bin_0<->bin_2"]
    assert summary["pair_max_force"]["bin_0<->bin_2"] == pytest.approx(2.5)
    assert summary["max_force_pair"] == "bin_0<->bin_2"
    assert summary["max_force_step"] == 10


def test_detect_swap_contacts_threshold_max_update_and_single_print(monkeypatch):
    printed = []
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: printed.append(args[0]))

    scene = _Scene(
        {
            ("bin_0", "bin_1"): (0.0, 0.0, swapContact.DEFAULT_SWAP_CONTACT_FORCE_EPS),
            ("bin_0", "bin_2"): (0.0, 3.0, 4.0),
        }
    )
    actors, swap_schedule, log_context, state = _build_inputs(scene)

    swapContact.detect_swap_contacts(
        scene=scene,
        actors=actors,
        swap_schedule=swap_schedule,
        timestep=10,
        state=state,
        log_context=log_context,
    )
    scene.forces[("bin_0", "bin_2")] = (0.0, 6.0, 8.0)
    swapContact.detect_swap_contacts(
        scene=scene,
        actors=actors,
        swap_schedule=swap_schedule,
        timestep=11,
        state=state,
        log_context=log_context,
    )

    summary = swapContact.get_swap_contact_summary(state)
    assert summary["contact_pairs"] == ["bin_0<->bin_2"]
    assert summary["pair_max_force"]["bin_0<->bin_2"] == pytest.approx(10.0)
    assert summary["max_force_norm"] == pytest.approx(10.0)
    assert summary["max_force_pair"] == "bin_0<->bin_2"
    assert summary["max_force_step"] == 11
    assert len(printed) == 1
    assert "episode=7 seed=13 step=10 pair=bin_0<->bin_2" in printed[0]


def test_reset_swap_contact_state_clears_previous_state():
    scene = _Scene({("bin_1", "bin_2"): (1.0, 2.0, 2.0)})
    actors, swap_schedule, log_context, state = _build_inputs(scene)

    swapContact.detect_swap_contacts(
        scene=scene,
        actors=actors,
        swap_schedule=swap_schedule,
        timestep=12,
        state=state,
        log_context=log_context,
    )
    assert swapContact.get_swap_contact_summary(state)["swap_contact_detected"] is True

    swapContact.reset_swap_contact_state(state)
    summary = swapContact.get_swap_contact_summary(state)
    assert summary == {
        "swap_contact_detected": False,
        "first_contact_step": None,
        "contact_pairs": [],
        "max_force_norm": 0.0,
        "max_force_pair": None,
        "max_force_step": None,
        "pair_max_force": {},
    }
    assert state.printed_pairs == set()
