import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

Q = 2013265921
Q_INV_NEG = 2013265919
W = 32
MASK = (1 << W) - 1


def montgomery_asr_reduce(t: int) -> int:
    m = ((t & MASK) * Q_INV_NEG) & MASK
    u_full = t + m * Q
    u_shift = (u_full >> W) & ((1 << (W + 1)) - 1)
    u_sub = u_shift - Q
    return u_shift if u_sub < 0 else u_sub


def add_mod_q(x: int, y: int) -> int:
    s = x + y
    return s - Q if s >= Q else s


def sub_mod_q(x: int, y: int) -> int:
    return x - y if x >= y else x + Q - y


@cocotb.test()
async def butterfly_stream_1_per_cycle(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    dut.rst_n.value = 0
    dut.in_valid.value = 0
    dut.coeff_a_i.value = 0
    dut.coeff_b_i.value = 0
    dut.twiddle_i.value = 0

    for _ in range(3):
        await RisingEdge(dut.clk)

    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    vectors = []
    expected = []
    for _ in range(16):
        a = random.randrange(0, Q)
        b = random.randrange(0, Q)
        w = random.randrange(0, Q)
        vectors.append((a, b, w))

        mr = montgomery_asr_reduce(b * w)
        expected.append((add_mod_q(a, mr), sub_mod_q(a, mr)))

    for a, b, w in vectors:
        dut.in_valid.value = 1
        dut.coeff_a_i.value = a
        dut.coeff_b_i.value = b
        dut.twiddle_i.value = w
        await RisingEdge(dut.clk)

    dut.in_valid.value = 0

    got = []
    for _ in range(24):
        await RisingEdge(dut.clk)
        if int(dut.out_valid.value) == 1:
            got.append((int(dut.coeff_a_o.value), int(dut.coeff_b_o.value)))

    assert len(got) == len(expected), f"expected {len(expected)} valid outputs, got {len(got)}"
    assert got == expected, "butterfly pipeline output mismatch"
