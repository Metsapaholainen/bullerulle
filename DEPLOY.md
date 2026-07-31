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
   ```
   This is the cloud equivalent of your local `.env` file -- `data/fmp_client.py`
   already checks `st.secrets["FMP_API_KEY"]` first and only falls back to
   `.env`/environment variables when no secrets are configured, so no code
   changes are needed to switch between local and cloud.

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
