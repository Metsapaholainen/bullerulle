# Deploying to Streamlit Community Cloud

This gives you a permanent URL for the scanner instead of starting a local
server each time. It's free, deploys straight from this GitHub repo, and
takes about 5 minutes. The account/connection step below is yours to do --
it can't be done on your behalf.

## Steps

1. **Push this repo to GitHub** (if you haven't already -- it should already
   be at `github.com/Metsapaholainen/bullerulle`). Confirm the two new
   preset files are committed: `config/preset_small_mid_cap.yaml` and
   `config/preset_large_cap.yaml` live in `config/`, which isn't gitignored,
   so a normal `git add`/`git commit`/`git push` picks them up automatically.

2. **Go to [share.streamlit.io](https://share.streamlit.io)** and sign in
   with your GitHub account.

3. Click **"New app"**, then pick:
   - Repository: `Metsapaholainen/bullerulle`
   - Branch: `master` (or whichever branch you want live)
   - Main file path: `app.py`

4. Before (or after) deploying, open **Advanced settings → Secrets** and add:
   ```toml
   FMP_API_KEY = "your_actual_fmp_api_key_here"
   NTFY_TOPIC = "your_private_ntfy_sh_topic_name_here"
   ```
   This is the cloud equivalent of your local `.env` file -- `data/fmp_client.py`
   already checks `st.secrets["FMP_API_KEY"]` first and only falls back to
   `.env`/environment variables when no secrets are configured, so no code
   changes are needed to switch between local and cloud. `NTFY_TOPIC` is the
   same idea for the Sell Alerts tab's push notifications (see below).

5. Click **Deploy**. First build takes a couple of minutes (installing
   `requirements.txt`).

## Things to know about the free tier

- **Ephemeral filesystem**: the container's disk (including the OHLCV
  parquet cache in `cache/` and the fundamentals JSON cache in
  `cache/fundamentals/`) does **not** persist across restarts -- every time
  the app sleeps from inactivity and wakes back up, or you push a new
  commit, it starts with an empty cache and re-fetches from FMP. This is an
  expected cost/latency tradeoff, not a bug. For a cloud deployment you'll
  mostly use, consider defaulting to a smaller custom symbol list or a
  lower "Max universe size" (sidebar → Universe filters) rather than the
  full ~1500-symbol auto-built universe, to keep load times reasonable.
- **Idle sleep**: free-tier apps go to sleep after a period of no visitors
  and take ~30-60 seconds to wake up on the next visit. This is normal.
- **Memory**: the free tier gives roughly 1GB of RAM. A modest universe
  (dozens to a few hundred symbols) with a couple years of daily history is
  fine; the full 1500-symbol universe with the widest history window may be
  tight -- if you hit memory errors, lower "Max universe size" first.

## Verifying it worked

Once deployed, load a small custom symbol list first (e.g. `NVDA,AMD,TSLA`)
before trying the full auto-built universe, to confirm the FMP key and data
pipeline work end-to-end on the cloud host before stressing it with a large
universe fetch.

## Sell Alerts: receiving the push notifications

Sell Alerts uses [ntfy.sh](https://ntfy.sh) -- free, no account needed. A
"topic" name is the only credential; anyone who knows it can read (and post
to) it, so pick something long and unguessable, not `test` or your name.

1. **Subscribe** to your topic (the same `NTFY_TOPIC` value used above) one
   of two ways:
   - Install the free **ntfy** app ([Android](https://play.google.com/store/apps/details?id=io.heckel.ntfy) /
     [iOS](https://apps.apple.com/us/app/ntfy/id1625396347)) and add your
     topic there for real push notifications on your phone.
   - Or just open `https://ntfy.sh/<your-topic>` in a browser tab and leave
     it open.
2. Use the **"Send test alert"** button on the app's Sell Alerts tab (or run
   `python cli.py sell-alerts --dry-run` locally to check the watchlist
   without sending) to confirm it arrives.

## Sell Alerts: the scheduled GitHub Actions check

The Sell Alerts tab only checks while you have the app open in a browser --
Streamlit Community Cloud can't run anything in the background. The real
"alert me even if I never open the app" path is a GitHub Actions workflow
(`.github/workflows/sell_alerts.yml`) that runs on its own schedule (weekday
afternoons, US market-close-ish) using GitHub's free infrastructure instead.

To turn it on:

1. In the GitHub repo, go to **Settings → Secrets and variables → Actions**
   and add two **repository secrets**:
   - `FMP_API_KEY` -- the same key you used above.
   - `NTFY_TOPIC` -- the same topic you subscribed to above.
2. That's it -- the workflow runs automatically on its schedule. To test it
   immediately rather than waiting: go to the repo's **Actions** tab, pick
   **"Sell alerts"**, and click **"Run workflow"** (this is what
   `workflow_dispatch` in the YAML enables).
3. Maintain the watchlist by editing `config/sell_watchlist.yaml` directly on
   GitHub (or editing it locally and pushing) -- it's a plain committed file,
   not something the scheduled job can write back to. Edits made in the
   Streamlit Cloud app's Sell Alerts tab don't persist for this job, since
   Cloud's container disk resets on every sleep/restart (see "Ephemeral
   filesystem" above) -- only edits made locally and pushed actually stick.
