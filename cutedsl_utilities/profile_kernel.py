#!/usr/bin/env python3
import argparse
import csv
import subprocess
import sys

def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile with Nsight Compute and annotate warnings per kernel block."
    )
    parser.add_argument("--ncu_path",        required=True, help="Path to the ncu executable")
    parser.add_argument("--exe_path",        required=True, help="Path to the interpreter or binary")
    parser.add_argument("--filepath",        required=True, help="Path to your script or binary to profile")
    parser.add_argument("--output_filename", required=True, help="Base name for the output CSV (no .csv)")
    parser.add_argument(
        "--filter",
        choices=["access", "stall", "warning"],
        help="If set, only keep lines matching that warning type"
    )
    return parser.parse_args()

def run_ncu(ncu_path, exe_path, filepath):
    cmd = [
        ncu_path,
        "--set", "full",
        "--csv",
        "--page", "source",
        "--print-source", "sass",
        "--force-overwrite",
        "--config-file", "off",
        exe_path,
        filepath
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, check=True)
    return proc.stdout.splitlines()

def strip_preamble(lines):
    for i, L in enumerate(lines):
        if L.startswith('"Kernel Name"'):
            return lines[i:]
    sys.exit("ERROR: 'Kernel Name' not found in Nsight output")

def split_blocks(lines):
    blocks, cur = [], []
    for L in lines:
        if L.startswith('"Kernel Name"'):
            if cur:
                blocks.append(cur)
            cur = [L]
        else:
            cur.append(L)
    if cur:
        blocks.append(cur)
    return blocks

def process_block(block_lines, filter_type):
    reader = csv.reader(block_lines)
    km_row = next(reader)
    rows   = list(reader)

    hdr_idx = next((i for i, r in enumerate(rows) if r and r[0] == "Address"), None)
    if hdr_idx is None:
        return [km_row]

    header_row = rows[hdr_idx]
    data_rows  = rows[hdr_idx + 1:]

    stall_idx      = header_row.index("Warp Stall Sampling (All Samples)")
    acc_excess_idx = header_row.index("L2 Theoretical Sectors Global Excessive")
    acc_total_idx  = header_row.index("L2 Theoretical Sectors Global")

    stall_sum = sum(int(r[stall_idx]) for r in data_rows if r[stall_idx].isdigit())

    out = []
    out.append(km_row)
    out.append(["#", "Warning Type", "Warning Info"] + header_row)

    annotated = []
    for orig_idx, r in enumerate(data_rows, start=1):
        wt, wi = [], []

        try:
            ex  = float(r[acc_excess_idx])
            tot = float(r[acc_total_idx])
            if ex > 0 and tot > 0 and ex >= 0.25 * tot:
                pct = (ex / tot) * 100
                wt.append("Access")
                wi.append(f"{pct:.2f}% of this line's global accesses are excessive")
        except:
            pass

        try:
            st = int(r[stall_idx])
            if stall_sum > 0 and st > 0 and st >= 0.10 * stall_sum:
                pct = (st / stall_sum) * 100
                wt.append("Stall")
                wi.append(f"This line is responsible for {pct:.1f}% of all warp stalls")
        except:
            pass

        annotated.append((orig_idx, r, wt, wi))

    if filter_type == "access":
        annotated = [(i, r, wt, wi) for (i, r, wt, wi) in annotated if "Access" in wt]
    elif filter_type == "stall":
        annotated = [(i, r, wt, wi) for (i, r, wt, wi) in annotated if "Stall" in wt]
    elif filter_type == "warning":
        annotated = [(i, r, wt, wi) for (i, r, wt, wi) in annotated if wt]

    for orig_idx, r, wt, wi in annotated:
        out.append([str(orig_idx), ", ".join(wt), " ".join(wi)] + r)

    return out

def main():
    args = parse_args()
    raw    = run_ncu(args.ncu_path, args.exe_path, args.filepath)
    kern   = strip_preamble(raw)
    blocks = split_blocks(kern)

    final = []
    for blk in blocks:
        final.extend(process_block(blk, args.filter))

    outpath = args.output_filename + ".csv"
    with open(outpath, "w", newline="") as f:
        csv.writer(f).writerows(final)

    print(f"Wrote annotated CSV to {outpath}")