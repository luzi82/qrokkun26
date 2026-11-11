import math
import numpy as np
import pytest
import torch
from qrokkun_ai.v24 import core as c


def test_birth_and_damped_gradient():
    x = torch.tensor([[0.0, 0.0], [1e-25, 1e-25], [-3.0, 4.0], [1e-6, 0.0]], dtype=torch.float64, requires_grad=True)
    a = c.birth_angle(x)
    assert a[0] == 0
    assert a[1].item() == pytest.approx(math.pi / 4)
    assert a[2].item() == pytest.approx(math.atan2(4, -3))
    a.sum().backward()
    assert torch.isfinite(x.grad).all()
    # Soft-damped gradient norm near zero is bounded by 1 / 0.25 = 4.0 (no 1e12 explosion!)
    assert x.grad[3].norm().item() <= 4.0 + 1e-6


def test_wrapped():
    grid = torch.linspace(-math.pi, math.pi, 40001, dtype=torch.float64)
    for sigma in (0.01, 0.1, 1.0, 3.0, math.pi):
        lp = c.wrapped_log_prob(grid, torch.tensor(0.72), torch.tensor(sigma))
        assert torch.trapezoid(lp.exp(), grid).item() == pytest.approx(1, abs=1e-10)
        edge = c.wrapped_log_prob(
            torch.tensor([-math.pi, math.pi], dtype=torch.float64), torch.tensor(0.72), torch.tensor(sigma)
        )
        assert edge[0].item() == pytest.approx(edge[1].item(), abs=1e-9)
    g = torch.Generator().manual_seed(17)
    a = c.wrapped_sample(torch.tensor(0.0), torch.tensor(3.0), g, (10000,))
    assert (a >= -math.pi).all() and (a < math.pi).all()


def test_speed():
    g = torch.Generator().manual_seed(17)
    for mu, s in [(30.0, 2.0), (115.0, 42.0), (200.0, 80.0)]:
        m = torch.tensor(mu, dtype=torch.float64, requires_grad=True)
        sigma = torch.tensor(s, dtype=torch.float64, requires_grad=True)
        a = c.speed_sample(m.expand(40000), sigma.expand(40000), g)
        assert (a >= 30).all() and (a <= 200).all()
        grid = torch.linspace(30, 200, 50001, dtype=torch.float64)
        p = c.speed_log_prob(grid, m, sigma).exp()
        assert torch.trapezoid(p, grid).item() == pytest.approx(1, abs=1e-5)
        c.speed_log_prob(a.detach(), m, sigma).mean().backward()
        assert torch.isfinite(m.grad) and torch.isfinite(sigma.grad)


def test_global_scales_and_bounded_fire_std():
    torch.manual_seed(17)
    net = c.Spawner()
    x = torch.stack([torch.zeros(325), torch.ones(325)])
    m, s = net.parameters_at(x)
    assert net.actor(x).shape == (2, 5) and s.shape == (2, 3)
    # State-independent global scales: identical across states!
    assert torch.equal(s[0], s[1])
    assert s[0].detach().numpy() == pytest.approx(list(c.STD_INIT), rel=1e-4)
    # Even under extreme logits, fire_std stays <= 0.25 rad (~14.3 deg)
    with torch.no_grad():
        net.scale_logits.fill_(100.0)
    assert net.global_scales()[2].item() == pytest.approx(c.STD_MAX[2], abs=1e-9)
    with torch.no_grad():
        net.scale_logits.fill_(0.0)
    _, s2 = net.parameters_at(x)
    s2.sum().backward()
    assert net.scale_logits.grad is not None and (net.scale_logits.grad.abs() > 0).all()


def test_conditional_and_pixel_lead_scale():
    torch.manual_seed(17)
    net = c.Spawner()
    x = torch.zeros(32, 325)
    x[:, 0:2] = 0.5
    g = torch.Generator().manual_seed(17)
    a, old = net.sample(x, g)
    new = net.log_prob(x, a)
    assert a.shape == (32, 3)
    assert torch.equal(old, new)
    assert torch.equal((new - old).exp(), torch.ones_like(new))
    # Verify unit change in m[..., 3:5] produces meaningful firing bearing change (~0.2+ rad)
    m0 = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    m1 = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
    f0 = c.fire_mean(x[:1], m0, torch.tensor([0.0], dtype=torch.float64))
    f1 = c.fire_mean(x[:1], m1, torch.tensor([0.0], dtype=torch.float64))
    assert abs(f1.item() - f0.item()) > 0.20
    (-new.mean() + net.value(x).square().mean()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())


def test_environment():
    from qrokkun_env.gv20 import spawn_frames, ray_position

    e = c.LearnedEnv(lambda obs: np.array([0.0, 115.0, 0.0]))
    e.reset(17)
    assert c.observation(e).shape == (325,)
    b = e.bullets[0]
    pos = (b.x, b.y)
    assert pos == ray_position(0.0) and b.radius == 2.6 and not b.moved
    e.step(0)
    assert (b.x, b.y) == pos
    for _ in range(119):
        e.step(0)
    assert [f for f, _ in e.s_history] == list(spawn_frames(121))
    for f, s in e.field_history:
        expected = np.mean([v for t, v in e.s_history if f - 60 <= t < f])
        assert s == expected
    assert e.field_history
    _, ret = c.timed_returns([0, 30, 60], 90)
    assert ret.tolist() == pytest.approx([-90 / 7200, -60 / 7200, -30 / 7200])
