# How to get a D-Wave Leap token (through Cineca)

The **token** is a password-like string that lets the code (`main_simulation.py qpu`) send problems to D-Wave's QPUs in the Leap cloud. You do not create it yourself. Every **Leap project** has one token, and you get access to a project that has QPU time **when Cineca's request (ISCRA-C) is approved**. A free Leap account alone has no usable token.

Overview:

```
1. Leap account          2. ISCRA-C request        3. approval            4. copy the token        5. give it to Ocean
 (institutional email) ─► (Cineca, D-Wave minutes) ─► (you join a Leap  ─► (Leap dashboard or   ─► dwave config create / dwave setup
                                                        project with time)   dwave CLI)               then: dwave solvers -la, dwave ping
```

> These steps follow Cineca's "Quantum Computing as a Service" page and D-Wave's Leap documentation (links at the bottom). Portals change: check the current call and deadline on Cineca's site before applying.

---

## Step 1: create a Leap account

1. Go to <https://cloud.dwavesys.com/leap/> and sign up.
2. **Use your institutional e-mail** (university or research institute). Cineca does not accept generic webmail such as gmail.
3. Confirm the e-mail. Every person who will use the QPU time needs **their own** Leap account.

At this point you can log in, but you have no QPU time and no usable token.

## Step 2: request D-Wave time from Cineca (ISCRA-C)

Cineca gives D-Wave time through the **ISCRA-C** calls, the same proposal system as its HPC machines. You submit them from the Cineca user portal (UserDB → ISCRA).

In the request:
* choose the **D-Wave / quantum** resource;
* list the **institutional e-mails of the Leap accounts** that will use the time (step 1);
* state the time in **minutes per month** and how to split it among the accounts.

According to Cineca's page, the maximum is 20 minutes/month for 3 months. Requests are evaluated every month. Unused monthly time is lost, and the service is closed in August.

**How much to ask for this project.** The `full` campaign uses about $2\times400\times0.155\ \text{s}=124\ \text{s}\approx2\ \text{min}$ of QPU time, for both topologies (see `explanation_project.pdf`, §4.5). Even with a chain-strength scan and a few repetitions, **5–10 minutes per month is ample**. Say in the proposal that the code was tested with simulators first (Cineca recommends this), and cite `README_CINECA.md`.

## Step 3: approval, and you join a Leap project

When the request is approved, your Leap account is added to a **Leap project** that carries the QPU minutes. Log in to Leap: the project appears in the project list (your **username menu → Projects**). If you belong to several projects, select the Cineca one: **each project has its own token**, and the dashboard shows the token of the *current* project.

## Step 4: copy the token

**Option A, in the browser (simplest):**
1. Log in at <https://cloud.dwavesys.com/leap/>.
2. Select the Cineca project (username menu → Projects).
3. On the dashboard, in the left column, find **API Token**. Click the eye icon to see it, or **COPY** to copy it.

**Option B, from a terminal** (also works on Leonardo, which has no browser):
```bash
source $WORK/venvs/skgame/bin/activate      # any environment with dwave-system installed
dwave auth login --oob                      # prints a link: open it on your laptop, log in, approve,
                                            # paste back the code that Leap shows you
dwave leap project ls                       # list your Leap projects; find the Cineca one
dwave leap project token --project <NAME>   # print that project's token
```

## Step 5: give the token to the code

Do this on every machine that will call the QPU (Leonardo, your laptop). Choose **one** of the following.

```bash
# (a) all in one: authorise, fetch the token of the chosen project, write the config file
dwave setup --oob --project <NAME>

# (b) paste it by hand
dwave config create          # at the prompt "Authentication token" paste it; press Enter for the rest
chmod 600 ~/.config/dwave/dwave.conf

# (c) environment variable (e.g. in ~/.bashrc; never in the repository)
export DWAVE_API_TOKEN=<token>
```

Then check:
```bash
dwave solvers -la            # should list an Advantage2_* (Zephyr) and an Advantage_* (Pegasus)
dwave ping                   # submits one tiny problem (a few ms of QPU time)
python main_simulation.py qpu --out <campaign dir> --check     # our check: solvers, clique sizes, estimated time; spends nothing
```

## Keep it safe

* The token gives access to **the project's QPU minutes**. Never commit it, paste it into notebooks, or share it.
* `~/.config/dwave/dwave.conf` should be readable only by you (`chmod 600`).
* If it leaks, click **RESET** on the Leap dashboard: a new token is issued and the old one stops working. Then repeat step 5.

## When something goes wrong

| symptom | likely cause |
|---|---|
| no project / no token on the dashboard | the ISCRA-C request is not approved yet, or was approved for a different e-mail than your Leap account |
| `SolverAuthenticationError`, HTTP 401 | wrong token, or the token of another project (step 3: select the Cineca project) |
| `dwave solvers -la` shows no Advantage2 or no Advantage | the project has no access to that QPU family: remove that back-end with `init --backends ... --force` |
| `dwave auth login` waits for a browser forever | use `--oob` (the link is opened on your laptop) |
| it works on the laptop but not on Leonardo | step 5 was not done on Leonardo, or the node has no internet (`cineca/check_internet.sh`) |

## Sources
* Cineca, Quantum Computing as a Service, how to access the quantum resources: <https://www.quantumcomputinglab.cineca.it/en/resources>
* D-Wave Support, How do I get an API token?: <https://support.dwavesys.com/hc/en-us/articles/360003682634>
* D-Wave docs, Authorizing access to the Leap service: <https://docs.dwavequantum.com/en/latest/ocean/leap_authorization.html>
* `dwave --help`, `dwave auth login --help`, `dwave setup --help` (dwave-cloud-client 0.14.8), for the commands above
