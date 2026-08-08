#!/bin/bash
# Tillicum first-login recon. Read-only, ~10 seconds, submits nothing and costs nothing.
#
# Everything in docs/tillicum.md marked UNVERIFIED was written from the public docs
# without an account. This settles those items in one pass so the doc, the dotfiles
# ssh target, and run_yolo_train_tillicum.slurm can be corrected from fact.
#
#   ./scripts/tillicum_recon.sh              (on a Tillicum login node)
#   wsl-ssh.ps1 tillicum script scripts/tillicum_recon.sh     (from a Claude session)
#
# CPU-only jobs are prohibited on Tillicum, so this deliberately runs on the LOGIN
# node -- it is all metadata queries, nothing that would warrant an allocation.

echo "=============================================================="
echo " Tillicum recon -- $(date)"
echo " host: $(hostname)   user: $USER"
echo "=============================================================="

echo
echo "## 1. HOME PATH  (the top unknown -- wsl-ssh.ps1 currently guesses)"
echo "home:      $(cd ~ && pwd)"
echo "quota:"
( quota -s 2>/dev/null || echo "  (quota not available)" ) | sed 's/^/  /'

echo
echo "## 2. PROJECT STORAGE"
for d in /gpfs/projects/makelab /gpfs/projects /gpfs; do
    if [ -d "$d" ]; then
        echo "$d exists; writable=$( [ -w "$d" ] && echo yes || echo NO )"
        df -h "$d" 2>/dev/null | tail -1 | sed 's/^/  /'
        break
    else
        echo "$d MISSING"
    fi
done
echo "contents of /gpfs/projects/makelab:"
ls -la /gpfs/projects/makelab 2>&1 | head -10 | sed 's/^/  /'

echo
echo "## 3. SCHEDULER  (expect: no partitions, QoS-driven)"
echo "-- sinfo --"
sinfo -o "%.14P %.6a %.10l %.6D %.6t %N" 2>&1 | head -10 | sed 's/^/  /'
echo "-- QoS available to us --"
sacctmgr -nP show assoc user="$USER" format=Account,QOS 2>&1 | sed 's/^/  /'
echo "-- do we already have long/wide/urgent? --"
sacctmgr -nP show assoc user="$USER" format=QOS 2>/dev/null | tr ',' '\n' \
    | grep -iE "long|wide|urgent" | sed 's/^/  /' || echo "  (none -- long QoS needs the request form)"

echo
echo "## 4. BILLING"
echo "-- hyakusage --"
( command -v hyakusage >/dev/null && hyakusage 2>&1 || echo "  hyakusage NOT on PATH" ) | sed 's/^/  /'

echo
echo "## 5. SOFTWARE  (docs say Apptainer, not modules -- verify)"
for t in apptainer singularity module conda python3 nvidia-smi rsync globus; do
    p=$(command -v "$t" 2>/dev/null)
    printf "  %-12s %s\n" "$t" "${p:-NOT FOUND}"
done
echo "-- apptainer version --"
( apptainer --version 2>&1 || echo "  n/a" ) | sed 's/^/  /'
echo "-- modules, if any --"
( module avail 2>&1 | head -15 || echo "  n/a" ) | sed 's/^/  /'

echo
echo "## 6. MAIL  (klone forced END,FAIL,TIME_LIMIT via a lua job_submit plugin and"
echo "##     ~1 email/min under preemption -- check whether Tillicum does the same)"
scontrol show config 2>/dev/null \
    | grep -iE "JobSubmitPlugins|MailProg|MailDomain|PreemptMode|MaxJobCount" | sed 's/^/  /'

echo
echo "## 7. NETWORK PATH BACK TO KLONE  (286 GB / ~911k files has to cross)"
echo "  NOTE: transfer ARCHIVES, not the tree. Per-file copy would pay ~911k round trips."
for h in klone.hyak.uw.edu; do
    printf "  %-24s " "$h"
    ( timeout 5 bash -c "</dev/tcp/$h/22" 2>/dev/null && echo "port 22 reachable" ) || echo "port 22 NOT reachable from here"
done

echo
echo "=============================================================="
echo " Next: update docs/tillicum.md (home path, QoS, mail, software),"
echo " fix Home in wsl-ssh.ps1 \$Targets, then run the debug smoke test."
echo "=============================================================="
