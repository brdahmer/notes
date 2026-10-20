#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qres.py -- per-residue fraction of native contacts, q_i(t), from an MD trajectory.
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap

import numpy as np

__version__ = "1.0"

# definitions per Best, Hummer & Eaton (2013)
DEF_BETA = 50.0     # nm^-1
DEF_LAMBDA = 1.8    # dimensionless
DEF_CUTOFF = 0.45   # nm
DEF_SEQSEP = 3      # contacts need |res_i - res_j| > SEQSEP


# --------------------------------------------------------------------------
# native contact set
# --------------------------------------------------------------------------

def find_native_contacts(ref_xyz, res_of_atom, cutoff, seqsep):
    n_atoms = len(ref_xyz)
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(ref_xyz)
        pairs = np.array(sorted(tree.query_pairs(cutoff)), dtype=np.int64)
        if pairs.size == 0:
            return (np.empty(0, np.int64), np.empty(0, np.int64),
                    np.empty(0, np.float64))
        ia, ja = pairs[:, 0], pairs[:, 1]
        keep = np.abs(res_of_atom[ia] - res_of_atom[ja]) > seqsep
        ia, ja = ia[keep], ja[keep]
        r0 = np.linalg.norm(ref_xyz[ia] - ref_xyz[ja], axis=1)
        return ia, ja, r0
    except ImportError:
        pass

    # ---- SciPy-free fallback: blocked pairwise distances -------------------
    block = 512
    pi, pj, r0 = [], [], []
    for a0 in range(0, n_atoms, block):
        a1 = min(a0 + block, n_atoms)
        d = np.linalg.norm(ref_xyz[a0:a1, None, :] - ref_xyz[None, :, :], axis=2)
        ia, ja = np.where(d < cutoff)
        ia = ia + a0
        keep = (ja > ia) & (np.abs(res_of_atom[ia] - res_of_atom[ja]) > seqsep)
        pi.append(ia[keep])
        pj.append(ja[keep])
        r0.append(d[ia[keep] - a0, ja[keep]])
    if not pi:
        return (np.empty(0, np.int64), np.empty(0, np.int64),
                np.empty(0, np.float64))
    return (np.concatenate(pi), np.concatenate(pj), np.concatenate(r0))


def switching(d, r0, beta, lam): #Best-Hummer-Eaton switching function
    arg = np.clip(beta * (d - lam * r0), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(arg))


def build_incidence(pi, pj, res_of_atom, n_res): #Map contacts -> residues.
    ri = res_of_atom[pi]
    rj = res_of_atom[pj]
    try:
        from scipy import sparse
        rows = np.concatenate([np.arange(len(pi)), np.arange(len(pi))])
        cols = np.concatenate([ri, rj])
        M = sparse.csr_matrix(
            (np.ones(len(rows), np.float64), (rows, cols)),
            shape=(len(pi), n_res))

        def accumulate(f):
            return np.asarray((M.T @ f.T).T)
    except ImportError:
        def accumulate(f):
            out = np.zeros((f.shape[0], n_res), np.float64)
            np.add.at(out, (slice(None), ri), f)
            np.add.at(out, (slice(None), rj), f)
            return out
    return accumulate


# --------------------------------------------------------------------------
# main computation
# --------------------------------------------------------------------------

def compute(traj_files, top_file, out_prefix, selection="protein",
            cutoff=DEF_CUTOFF, seqsep=DEF_SEQSEP, beta=DEF_BETA,
            lam=DEF_LAMBDA, stride=1, chunk=100, ref_file=None, ref_frame=0,
            periodic=True, time_unit="ns", npz=False, verbose=True):
    import mdtraj as md

    say = print if verbose else (lambda *a, **k: None)

    # ---- topology and atom selection --------------------------------------
    top = md.load_topology(top_file)
    sel_user = top.select(selection)
    if len(sel_user) == 0:
        raise SystemExit(f"selection '{selection}' matched no atoms in {top_file}")
    # select heavy atoms
    heavy_all = set(top.select_atom_indices("heavy").tolist())
    sel = np.array(sorted(set(sel_user.tolist()) & heavy_all), dtype=np.int64)
    if len(sel) == 0:
        raise SystemExit(f"selection '{selection}' contains no heavy atoms")

    residues = sorted({top.atom(int(i)).residue for i in sel},
                      key=lambda r: r.index)
    res_pos = {r.index: k for k, r in enumerate(residues)}
    n_res = len(residues)
    res_of_atom = np.array([res_pos[top.atom(int(i)).residue.index] for i in sel],
                           dtype=np.int64)
    # labels use the numbering in the structure file
    resids = np.array([r.resSeq for r in residues], dtype=np.int64)
    resnames = [r.name for r in residues]
    chains = [r.chain.index for r in residues]

    say(f"  system      : {len(sel)} heavy atoms in {n_res} residues "
        f"(selection: {selection})")

    # ---- reference state --------------------------------------------------
    if ref_file:
        try:                       # pdb/gro/h5 carry their own topology
            ref = md.load(ref_file, atom_indices=sel)
        except (TypeError, ValueError):
            ref = md.load(ref_file, top=top_file, atom_indices=sel)
        ref_xyz = ref.xyz[0].astype(np.float64)
        say(f"  reference   : {os.path.basename(ref_file)} (frame 0)")
    else:
        ref = md.load_frame(traj_files[0], ref_frame, top=top_file,
                            atom_indices=sel)
        ref_xyz = ref.xyz[0].astype(np.float64)
        say(f"  reference   : frame {ref_frame} of "
            f"{os.path.basename(traj_files[0])}")

    # ---- native contact set ----------------------------------------------
    pi, pj, r0 = find_native_contacts(ref_xyz, res_of_atom, cutoff, seqsep)
    n_pairs = len(pi)
    if n_pairs == 0:
        raise SystemExit(
            "no native contacts found -- check --selection, --cutoff and that "
            "the reference structure is not garbled")
    n_a = np.zeros(n_res, np.float64)
    np.add.at(n_a, res_of_atom[pi], 1.0)
    np.add.at(n_a, res_of_atom[pj], 1.0)
    say(f"  native set  : |S| = {n_pairs} contacts  "
        f"(cutoff {cutoff*10:.1f} A, |i-j| > {seqsep})")
    say(f"                sum(n_a) = {int(n_a.sum())} = 2|S| "
        f"({'ok' if int(n_a.sum()) == 2 * n_pairs else 'MISMATCH'}); "
        f"{int((n_a == 0).sum())} residues have none (reported as NaN)")

    accumulate = build_incidence(pi, pj, res_of_atom, n_res)
    r0_row = r0[None, :]

    # ---- sweep the trajectory in chunks -----------------------------------
    num_blocks, q_direct_blocks, times = [], [], []
    n_frames = 0
    for tf in traj_files:
        for tr in md.iterload(tf, top=top_file, chunk=chunk, stride=stride,
                              atom_indices=sel):
            d = md.compute_distances(tr, np.column_stack([pi, pj]),
                                     periodic=periodic)
            f = switching(d.astype(np.float64), r0_row, beta, lam)
            q_direct_blocks.append(f.mean(axis=1))
            num_blocks.append(accumulate(f))
            times.append(np.asarray(tr.time, dtype=np.float64))
            n_frames += len(tr)
            if verbose:
                sys.stdout.write(f"\r  frames read : {n_frames}      ")
                sys.stdout.flush()
    if verbose:
        sys.stdout.write("\n")
    if n_frames == 0:
        raise SystemExit("no frames were read -- check the trajectory files")

    num = np.concatenate(num_blocks, axis=0)
    q_direct = np.concatenate(q_direct_blocks)
    t = np.concatenate(times)
    del num_blocks, q_direct_blocks

    # MDTraj reports time in ps. If the file carried no time information the
    # frame index is used instead, which is stated in the CSV header.
    have_time = np.any(np.diff(t) != 0) if len(t) > 1 else bool(t[0])
    if not have_time:
        t = np.arange(n_frames, dtype=np.float64) * stride
        t_label, t_out = "frame", t
    elif time_unit == "ns":
        t_label, t_out = "time_ns", t / 1000.0
    else:
        t_label, t_out = "time_ps", t

    # ---- q_a and reconstruction of global Q ----------------------------------
    has = n_a > 0
    q_res = np.full((n_frames, n_res), np.nan)
    q_res[:, has] = num[:, has] / n_a[None, has]
    q_rebuilt = np.nansum(q_res[:, has] * n_a[None, has], axis=1) / n_a.sum()

    resid = float(np.max(np.abs(q_direct - q_rebuilt)))
    say(f"  Q(t)        : mean {q_direct.mean():.4f}  sd {q_direct.std():.4f}  "
        f"min {q_direct.min():.4f}")
    say(f"  consistency : max |Q_direct - Q_from_q_a| = {resid:.2e} "
        f"({'ok' if resid < 1e-6 else 'CHECK THIS'})")

    # ---- write ------------------------------------------------------------
    outdir = os.path.dirname(out_prefix)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    hdr_common = (
        f"# qres.py v{__version__} -- per-residue fraction of native contacts\n"
        f"# Best, Hummer & Eaton, PNAS 110:17874 (2013)\n"
        f"# topology={os.path.basename(top_file)} "
        f"traj={','.join(os.path.basename(x) for x in traj_files)}\n"
        f"# selection='{selection}' heavy_atoms={len(sel)} residues={n_res}\n"
        f"# cutoff={cutoff} nm  seqsep={seqsep}  beta={beta} nm^-1  "
        f"lambda={lam}  stride={stride}  periodic={periodic}\n"
        f"# reference={'file:' + os.path.basename(ref_file) if ref_file else 'frame ' + str(ref_frame)}"
        f"  n_native_contacts={n_pairs}\n")

    p_q = out_prefix + "_qres.csv"
    with open(p_q, "w") as fh:
        fh.write(hdr_common)
        fh.write("# columns after the first are residues, labelled by resSeq; "
                 "NaN = residue has no native contacts\n")
        fh.write(t_label + "," + ",".join(str(r) for r in resids) + "\n")
        for k in range(n_frames):
            row = ",".join("nan" if np.isnan(v) else f"{v:.5f}"
                           for v in q_res[k])
            fh.write(f"{t_out[k]:.4f},{row}\n")

    # statistics only over residues that actually form contacts
    q_mean = np.full(n_res, np.nan)
    q_sd = np.full(n_res, np.nan)
    q_min = np.full(n_res, np.nan)
    q_mean[has] = q_res[:, has].mean(axis=0)
    q_sd[has] = q_res[:, has].std(axis=0)
    q_min[has] = q_res[:, has].min(axis=0)
    p_s = out_prefix + "_qres_summary.csv"
    with open(p_s, "w") as fh:
        fh.write(hdr_common)
        fh.write("resid,resname,chain,n_native_contacts,q_mean,q_sd,q_min\n")
        for k in range(n_res):
            if n_a[k] == 0:
                fh.write(f"{resids[k]},{resnames[k]},{chains[k]},0,nan,nan,nan\n")
            else:
                fh.write(f"{resids[k]},{resnames[k]},{chains[k]},"
                         f"{int(n_a[k])},{q_mean[k]:.5f},{q_sd[k]:.5f},"
                         f"{q_min[k]:.5f}\n")

    p_g = out_prefix + "_Q_global.csv"
    with open(p_g, "w") as fh:
        fh.write(hdr_common)
        fh.write("# Q_direct   = mean of f over all native contacts\n"
                 "# Q_from_qres= sum(n_a * q_a) / sum(n_a), the contact-weighted\n"
                 "#              mean; equals Q_direct by construction. The\n"
                 "#              UNWEIGHTED mean over residues does not.\n")
        fh.write(f"{t_label},Q_direct,Q_from_qres,abs_diff\n")
        for k in range(n_frames):
            fh.write(f"{t_out[k]:.4f},{q_direct[k]:.6f},{q_rebuilt[k]:.6f},"
                     f"{abs(q_direct[k]-q_rebuilt[k]):.2e}\n")

    say(f"  written     : {os.path.basename(p_q)}, "
        f"{os.path.basename(p_s)}, {os.path.basename(p_g)}")

    if npz:
        p_n = out_prefix + "_qres.npz"
        np.savez_compressed(
            p_n, resids=resids, q_res=q_res.astype(np.float32),
            q_direct=q_direct, q_rebuilt=q_rebuilt, n_a=n_a, time=t_out,
            time_label=np.array(t_label), n_native_contacts=n_pairs,
            resnames=np.array(resnames), chains=np.array(chains),
            params=np.array([cutoff, seqsep, beta, lam]))
        say(f"                {os.path.basename(p_n)} "
            f"(same data, compressed; np.load(...)['q_res'])")

    return dict(resids=resids, q_res=q_res, q_direct=q_direct,
                q_rebuilt=q_rebuilt, n_a=n_a, t=t_out, t_label=t_label,
                n_pairs=n_pairs, q_mean=q_mean, q_sd=q_sd)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def make_plots(res, out_prefix, verbose=True):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  plots skipped: matplotlib is not installed "
              "(pip install matplotlib)")
        return

    resids, q_res, t = res["resids"], res["q_res"], res["t"]
    t_label = res["t_label"].replace("_", " (") + (
        ")" if res["t_label"] != "frame" else "")

    # --- heatmap: residue x time ------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.0), constrained_layout=True)
    # viridis is perceptually uniform and readable in greyscale and for
    # colour-vision deficiency; residues with no contacts stay white.
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    # the y axis is residue ORDER, not residue number, so gaps in the
    # numbering (missing loops, several chains) do not stretch the picture;
    # the ticks carry the real resSeq labels.
    im = ax.imshow(np.ma.masked_invalid(q_res).T, aspect="auto",
                   origin="lower", cmap=cmap, vmin=0.0, vmax=1.0,
                   extent=[t[0], t[-1], -0.5, len(resids) - 0.5],
                   interpolation="nearest")
    step = max(1, len(resids) // 12)
    ticks = np.arange(0, len(resids), step)
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(resids[i]) for i in ticks])
    ax.set_xlabel(t_label)
    ax.set_ylabel("residue")
    ax.set_title("per-residue fraction of native contacts, $q_a(t)$")
    cb = fig.colorbar(im, ax=ax, pad=0.015)
    cb.set_label("$q_a$")
    p1 = out_prefix + "_qres_heatmap.png"
    fig.savefig(p1, dpi=200)
    plt.close(fig)

    # --- profile: time-averaged q per residue ------------------------------
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(9, 5.4), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 1]})
    mu, sd = res["q_mean"], res["q_sd"]
    # q is bounded on [0, 1], so the band is clipped rather than drawn
    # outside the range the quantity can take
    ax.fill_between(resids, np.clip(mu - sd, 0, 1), np.clip(mu + sd, 0, 1),
                    alpha=0.25, linewidth=0, color="#3b6ea5",
                    label="$\\pm$1 sd over frames")
    ax.plot(resids, mu, lw=1.1, color="#1f3f66", label="mean $q_a$")
    ax.axhline(np.nanmean(res["q_direct"]), ls="--", lw=0.9, color="#b1400a",
               label=f"global Q = {np.nanmean(res['q_direct']):.3f}")
    ax.set_ylabel("$q_a$")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("time-averaged native-contact retention per residue")

    ax2.bar(resids, res["n_a"], width=1.0, color="#9aa7b4", linewidth=0)
    ax2.set_ylabel("$n_a$")
    ax2.set_xlabel("residue")
    ax2.text(0.005, 0.86, "native contacts per residue -- residues with a "
                          "small $n_a$ have a coarse, noisy $q_a$",
             transform=ax2.transAxes, fontsize=7.5, color="#4a5560")
    for a in (ax, ax2):
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    p2 = out_prefix + "_qres_profile.png"
    fig.savefig(p2, dpi=200)
    plt.close(fig)

    if verbose:
        print(f"  figures     : {os.path.basename(p1)}, {os.path.basename(p2)}")


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

def selftest():
    """Check the decomposition on a case small enough to verify by hand."""
    print("=" * 74)
    print("SELF-TEST -- 5 contacts over 6 residues")
    print("=" * 74)
    # contact -> (residue a, residue b, f value at this frame)
    contacts = [(1, 5, 0.97), (1, 5, 0.95), (1, 5, 0.96),
                (1, 5, 0.98), (2, 4, 0.10)]
    f = np.array([c[2] for c in contacts])
    n_res = 6
    pi = np.arange(0, 2 * len(contacts), 2)
    pj = pi + 1
    lut = np.zeros(2 * len(contacts), np.int64)
    for k, (a, b, _) in enumerate(contacts):
        lut[2 * k], lut[2 * k + 1] = a, b
    acc = build_incidence(pi, pj, lut, n_res)
    num = acc(f[None, :])[0]
    den = np.zeros(n_res)
    np.add.at(den, lut[pi], 1.0)
    np.add.at(den, lut[pj], 1.0)
    has = den > 0
    q = np.full(n_res, np.nan)
    q[has] = num[has] / den[has]

    Q_direct = f.mean()
    Q_weighted = np.nansum(q[has] * den[has]) / den.sum()
    Q_plain = np.nanmean(q[has])

    print("  residue   n_a    q_a")
    for a in range(n_res):
        if has[a]:
            print(f"    {a}       {int(den[a])}     {q[a]:.4f}")
    print(f"\n  Q direct (mean of the {len(f)} f values)   = {Q_direct:.6f}")
    print(f"  Q rebuilt, contact-weighted             = {Q_weighted:.6f}"
          f"   diff {abs(Q_direct - Q_weighted):.1e}")
    print(f"  plain mean over residues (WRONG)        = {Q_plain:.6f}"
          f"   off by {Q_plain - Q_direct:+.4f}")
    print(f"  sum(n_a) = {int(den.sum())} = 2 x {len(f)} contacts   "
          f"{'ok' if int(den.sum()) == 2 * len(f) else 'MISMATCH'}")

    # switching function sanity check
    r0 = np.array([0.40])
    assert switching(np.array([0.40]), r0, DEF_BETA, DEF_LAMBDA)[0] > 0.98
    assert switching(np.array([2.00]), r0, DEF_BETA, DEF_LAMBDA)[0] < 1e-20
    assert abs(switching(np.array([DEF_LAMBDA * 0.40]), r0,
                         DEF_BETA, DEF_LAMBDA)[0] - 0.5) < 1e-12
    ok = abs(Q_direct - Q_weighted) < 1e-12
    print(f"\n  switching function: f(r0)~1, f(lambda*r0)=0.5, f(far)~0   ok")
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="qres.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Per-residue fraction of native contacts q_a(t) from an MD
            trajectory, using the Best-Hummer-Eaton definition (the same one
            behind MDTraj's global best_hummer_q). Global Q(t) is written
            alongside, both computed directly and rebuilt from the per-residue
            terms as a check."""),
        epilog=textwrap.dedent("""\
            examples:
              python qres.py -f md.xtc -s md.gro -o output
              python qres.py -f p1.xtc p2.xtc -s md.tpr --stride 10 -o output --plot
              python qres.py --selftest

            notes:
              * distances are in nanometres throughout (MDTraj's unit); pass
                --cutoff 0.45 for 4.5 A.
              * sequence separation uses residue ORDER in the topology, so it
                also excludes near-neighbours across a chain break. Contacts
                between different chains are always kept.
              * the trajectory does not need to be fitted: q depends only on
                internal distances. It DOES need to be whole (molecules not
                broken across the periodic box) -- e.g.
                `gmx trjconv -pbc mol -ur compact`. With --nopbc the minimum
                image convention is switched off entirely.
            """))
    ap.add_argument("-f", "--traj", nargs="+", metavar="FILE",
                    help="trajectory file(s): xtc, trr, dcd, nc, h5, ...")
    ap.add_argument("-s", "--top", metavar="FILE",
                    help="topology/structure: gro, pdb, prmtop, psf, ...")
    ap.add_argument("-o", "--out", default="qres", metavar="PREFIX",
                    help="output prefix (default: qres)")
    ap.add_argument("--selection", default="protein", metavar="STR",
                    help="MDTraj atom selection; heavy atoms are taken from it "
                         "(default: protein)")
    ap.add_argument("--ref", metavar="FILE", default=None,
                    help="use this structure as the native state instead of a "
                         "trajectory frame (e.g. the energy-minimised model)")
    ap.add_argument("--ref-frame", type=int, default=0, metavar="N",
                    help="trajectory frame used as the native state "
                         "(default: 0)")
    ap.add_argument("--cutoff", type=float, default=DEF_CUTOFF, metavar="NM",
                    help=f"native contact cutoff in nm (default: {DEF_CUTOFF})")
    ap.add_argument("--seqsep", type=int, default=DEF_SEQSEP, metavar="N",
                    help=f"require |i-j| > N residues (default: {DEF_SEQSEP})")
    ap.add_argument("--beta", type=float, default=DEF_BETA, metavar="X",
                    help=f"switching steepness, nm^-1 (default: {DEF_BETA})")
    ap.add_argument("--lam", "--lambda", dest="lam", type=float,
                    default=DEF_LAMBDA, metavar="X",
                    help=f"contact tolerance factor (default: {DEF_LAMBDA})")
    ap.add_argument("--stride", type=int, default=1, metavar="N",
                    help="use every Nth frame (default: 1)")
    ap.add_argument("--chunk", type=int, default=100, metavar="N",
                    help="frames held in memory at once; lower it if you run "
                         "out of RAM (default: 100)")
    ap.add_argument("--nopbc", action="store_true",
                    help="do not apply the minimum image convention")
    ap.add_argument("--time-unit", choices=["ns", "ps"], default="ns",
                    help="time column in the output (default: ns)")
    ap.add_argument("--npz", action="store_true",
                    help="also write everything as a compressed .npz; worth it "
                         "for long trajectories, where the wide CSV gets big")
    ap.add_argument("--plot", action="store_true",
                    help="also write a heatmap and a per-residue profile")
    ap.add_argument("-q", "--quiet", action="store_true")
    ap.add_argument("--selftest", action="store_true",
                    help="verify the maths on a synthetic case and exit")
    ap.add_argument("--version", action="version",
                    version=f"qres.py {__version__}")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()
    if not args.traj or not args.top:
        ap.error("-f/--traj and -s/--top are required (or use --selftest)")
    for p in list(args.traj) + [args.top] + ([args.ref] if args.ref else []):
        if not os.path.exists(p):
            ap.error(f"file not found: {p}")

    try:
        import mdtraj  # noqa: F401
    except ImportError:
        raise SystemExit("mdtraj is not installed.  pip install mdtraj\n"
                         "(or: conda install -c conda-forge mdtraj)")

    verbose = not args.quiet
    if verbose:
        print("=" * 74)
        print(f"qres.py v{__version__} -- per-residue fraction of native contacts")
        print("=" * 74)

    res = compute(
        traj_files=args.traj, top_file=args.top, out_prefix=args.out,
        selection=args.selection, cutoff=args.cutoff, seqsep=args.seqsep,
        beta=args.beta, lam=args.lam, stride=args.stride, chunk=args.chunk,
        ref_file=args.ref, ref_frame=args.ref_frame,
        periodic=not args.nopbc, time_unit=args.time_unit, npz=args.npz,
        verbose=verbose)

    if args.plot:
        make_plots(res, args.out, verbose=verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
