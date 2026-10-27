# Eseguire `main_simulation` su Cineca (Leonardo) e sul QPU D-Wave (Zephyr e Pegasus)

Ipotesi (cambiale se non valgono, vedi § 13):
* la macchina è **Leonardo**;
* il progetto Leap di Cineca è **già approvato** e hai il suo **token API**. Senza token puoi fare tutto tranne gli stadi `qpu`: usa `sim`.

Il QPU non è a Bologna: è nel cloud Leap di D-Wave. Per usarlo servono **solo un token e HTTPS in uscita**, da qualunque macchina. Per questo lo stadio QPU si può lanciare **da Leonardo** (nodo di login o partizione seriale) **oppure dal tuo computer** (§ 8).

---

## 0. Cosa fa la simulazione

Per ogni **regime** e ogni **N** su una griglia lineare (preset `full`: N = 8, 16, …, 160), sulle **stesse** istanze J:

| regime | ε | J0 | α | β | problema del planner | fase del gioco | atteso |
|---|---|---|---|---|---|---|---|
| `ferro_easy` | 0 | 3 | 0.1 | ∞ | ferromagnete | punti fissi | ottimo facile per tutti |
| `sg_boundary` | 0 | 1 | 0.1 | ∞ | bordo SG/ferro | punti fissi | intermedio |
| `sg_satisficing` | 0 | 0 | 0.1 | ∞ | spin glass | punti fissi | ottimo difficile; il gioco si ferma sotto l'ottimo |
| `sg_cycles` | 0 | 0 | 1 | ∞ | spin glass | 2-cicli | il gioco non lo trova mai |
| `nonreciprocal_chaos` | 1 | 0 | 0.1 | ∞ | spin glass | caotica | il gioco non lo trova mai |
| `paramagnet` | 0 | 0 | 0.1 | 0.25 | spin glass | paramagnete (T=4 > T_c=2) | gioco casuale |

Su ogni istanza si calcolano:
* ottimo di riferimento (esatto per N ≤ 24, migliore noto sopra);
* solver classici;
* il **gioco**;
* il **quantum annealer** con due back-end, sulle stesse istanze e con le stesse impostazioni:

| back-end | QPU | grafo | embedding (all-to-all) | catene a N = 160 |
|---|---|---|---|---|
| `qpu:zephyr:clique` | Advantage2 | Zephyr, grado 20 | clique nativo (busclique) | 12 qubit |
| `qpu:pegasus:clique` | Advantage | Pegasus, grado 15 | clique nativo (busclique) | 15 qubit |

L'analisi dice quale topologia trova l'ottimo più spesso, per ogni N e classe di problema. Come funzionano embedding e catene, con le figure passo passo: `../visualization/embedding_explanation/`.

Stadi **indipendenti e riprendibili** (un file già scritto non viene ricalcolato):

| stadio | comando | dove | internet? |
|---|---|---|---|
| init | `python main_simulation.py init --preset full --out DIR` | login o locale | no |
| classico + gioco | `python main_simulation.py classical --out DIR --task i` | nodi di calcolo (array, i = 0..3) | no |
| embedding | `python main_simulation.py embed --out DIR` | dove c'è HTTPS (legge il grafo vero del QPU) | **sì** |
| QPU | `python main_simulation.py qpu --out DIR` | dove c'è HTTPS: `lrd_all_serial`, login o **il tuo computer** | **sì** |
| analisi | `python main_simulation.py analyze --out DIR` | ovunque | no |

---

## 1. Login

```bash
step ssh login '<tua-email>' --provisioner cineca-hpc
ssh <username>@login.leonardo.cineca.it
```

## 2. Codice: clone (una volta) e pull (ogni aggiornamento)

```bash
saldo -b                                   # nome del tuo account/budget (colonna "account")
echo $WORK                                 # se vuota: export WORK=/leonardo_work/<ACCOUNT>
cd $WORK
git clone https://github.com/albertoamaduzzi/Projects_master_HPQC.git
cd $WORK/Projects_master_HPQC && git pull  # aggiornamenti successivi
cd $WORK/Projects_master_HPQC/scripts/QuantumComputing/Dimod/src/unlearnable_games_explanation
```
(Se il repository è privato, GitHub chiede username + personal access token.)

## 3. Ambiente Python: costruito sul tuo computer e caricato su Leonardo

Su Leonardo non si scaricano pacchetti: l'ambiente si prepara fuori e si carica come archivio di *wheel*. Il pacchetto `skgame` non va installato, perché gli script girano dalla cartella del codice.

**3a. Su Leonardo**: leggi la versione di Python del modulo.
```bash
module load python && python --version     # es. "Python 3.11.7" -> la versione da usare è 3.11
```

**3b. Sul tuo computer** (serve solo `python3` con `pip`, su Linux, macOS o WSL; dalla cartella `unlearnable_games_explanation` del repository):
```bash
bash cineca/build_wheelhouse.sh 3.11                 # -> skgame_wheelhouse_py311.tar.gz (~185 MB)
WITH_NB=1 bash cineca/build_wheelhouse.sh 3.11       # in alternativa: + jupyter, se vuoi eseguire il notebook su Leonardo
scp skgame_wheelhouse_py311.tar.gz <username>@login.leonardo.cineca.it:/leonardo_work/<ACCOUNT>/
```
Lo script scarica le wheel per Linux x86_64 (glibc ≥ 2.17/2.28) e per quella versione di Python, qualunque sia il tuo sistema operativo. Scrive anche `requirements.lock` con le versioni esatte.

**3c. Su Leonardo**: installazione senza internet.
```bash
bash cineca/install_offline.sh $WORK/skgame_wheelhouse_py311.tar.gz   # crea $WORK/venvs/skgame
source $WORK/venvs/skgame/bin/activate
pytest -q                                                            # 107 test, nessun token (~1 min)
```
Se la versione di Python non coincide, lo script si ferma e dice quale wheelhouse ricostruire. L'ho provato così: wheelhouse costruito e installato con `--no-index` (nessun download), poi 107 test passati.

## 4. Configura `cineca/env.sh` (una volta)

```bash
export SKGAME_ACCOUNT="${SKGAME_ACCOUNT:-CHANGE_ME}"          # -> il tuo account di `saldo -b`
export SKGAME_PARTITION="${SKGAME_PARTITION:-dcgp_usr_prod}"  # partizione CPU; se il budget è solo Booster: boost_usr_prod
export SKGAME_VENV="${SKGAME_VENV:-$WORK/venvs/skgame}"
export SKGAME_OUT="${SKGAME_OUT:-$WORK/skgame_results/full}"
```
Poi `source cineca/env.sh`, che carica il modulo, il venv e le variabili. Ogni job lo fa da solo.

## 5. Token Leap (dove girerà lo stadio QPU)

Come ottenerlo passo per passo (account Leap, richiesta ISCRA-C a Cineca, copia del token, `dwave setup --oob` su Leonardo senza browser): [`../../LEAP_TOKEN.md`](../../LEAP_TOKEN.md).

Il token si copia dalla dashboard Leap (https://cloud.dwavesys.com/leap/), **selezionando il progetto Cineca**: il token è per progetto.
```bash
dwave config create          # alla domanda "Authentication token" incolla il token; il resto: Invio
chmod 600 ~/.config/dwave/dwave.conf
dwave solvers -la            # devono comparire un Advantage2_* (Zephyr) e un Advantage_* (Pegasus)
dwave ping                   # spende pochi ms di QPU
```
In alternativa `export DWAVE_API_TOKEN=...` (in `~/.bashrc`, **mai** nel repository). Su Leonardo la `$HOME` è condivisa con i job.

## 6. Dove c'è internet su Leonardo?

```bash
bash cineca/check_internet.sh
```

| risultato | stadio QPU |
|---|---|
| `lrd_all_serial` raggiungibile | `sbatch -A $SKGAME_ACCOUNT cineca/job_qpu.sbatch` (già in `submit_all.sh`) |
| solo login raggiungibile | `bash cineca/run_qpu_login.sh` (in background con `nohup`) |
| nessuno | dal tuo computer (§ 8) |

`lrd_all_serial` è la partizione seriale che gira sui nodi di login: max 4 h, 4 core, non scala il budget. Controlla con `sinfo -p lrd_all_serial`.

## 7. Prova a vuoto (2 minuti, nessun token)

```bash
source cineca/env.sh
D=$WORK/skgame_results/smoke
python main_simulation.py init --preset smoke --out $D --backends qpu:zephyr:clique qpu:pegasus:clique sim:zephyr:clique sim:pegasus:clique
python main_simulation.py classical --out $D
python main_simulation.py sim --out $D --backend sim:logical sim:zephyr:clique sim:pegasus:clique   # annealing classico, catene incluse
python main_simulation.py analyze   --out $D
ls $D/summary/      # summary.csv, backend_comparison.csv, figure .png
```
`sim:zephyr:clique` e `sim:pegasus:clique` risolvono lo **stesso problema con le catene**, ma con simulated annealing classico sul grafo ideale. Servono a provare tutta la pipeline senza token.

## 8. Lanciare il QPU dal tuo computer (Leap in locale)

Sì, si può: lo stadio QPU legge solo `config.json` e `instances/`, chiama Leap via HTTPS e scrive `anneal/` e `embeddings/`. Flusso consigliato, con la parte classica su Leonardo e il QPU in locale:

```bash
# --- sul tuo computer, nella cartella unlearnable_games_explanation del repository ---
python3 -m venv .venv && source .venv/bin/activate
pip install -r cineca/requirements.txt
dwave config create                                            # token del progetto Cineca

export LEONARDO=<username>@login.leonardo.cineca.it
export REMOTE_OUT=/leonardo_work/<ACCOUNT>/skgame_results/full # = $SKGAME_OUT su Leonardo
export LOCAL_OUT=results/full

bash cineca/sync.sh pull-inputs                                # config.json + instances/ da Leonardo
python main_simulation.py qpu --out $LOCAL_OUT --check         # solver, clique max, stima: non spende QPU
python main_simulation.py qpu --out $LOCAL_OUT --max-calls 2   # due chiamate vere per controllare
python main_simulation.py qpu --out $LOCAL_OUT                 # tutto (Zephyr poi Pegasus); riprendibile
bash cineca/sync.sh push-anneal                                # anneal/ + embeddings/ su Leonardo
```
Prima di `pull-inputs` su Leonardo devono esistere `config.json` (§ 9, `init`) e le istanze. Le istanze le crea anche il job classico; per averle subito, lancia su Leonardo `python main_simulation.py instances --out $SKGAME_OUT` (pochi secondi, senza internet).

Si può fare anche al contrario: `init` e QPU in locale, poi `sync.sh push-inputs` e stadio classico su Leonardo. Le istanze sono deterministiche (seed in `config.json`) e una volta scritte vengono sempre riusate.

## 9. Campagna completa su Leonardo

```bash
source cineca/env.sh
python main_simulation.py init --preset full --out $SKGAME_OUT     # config.json + piano (back-end, budget)
python main_simulation.py embed --out $SKGAME_OUT                  # embedding Zephyr e Pegasus sui grafi veri (serve token, 0 QPU)
python main_simulation.py qpu --out $SKGAME_OUT --check            # stima del QPU per ogni solver
bash cineca/submit_all.sh                                          # array classico + job QPU + analisi in coda
```
`submit_all.sh` sottomette:
1. `job_classical.sbatch`: array `0-3`, una classe di problema per task, 6 h;
2. `job_qpu.sbatch`: su `lrd_all_serial`, esegue **entrambi** i back-end QPU. Se non c'è internet lì: `bash cineca/submit_all.sh --no-qpu` e poi `run_qpu_login.sh` oppure § 8;
3. `job_analyze.sbatch`: parte quando 1 e 2 sono finiti.

```bash
squeue --me
tail -f logs/skgame-classical_*_0.out logs/skgame-qpu_*.out
```
Se un job finisce per walltime o cade la rete, **risottometti lo stesso comando**: riparte da dove era.

Varianti di `init` (`--force` per sovrascrivere un config esistente):
```bash
python main_simulation.py init --preset full --out $SKGAME_OUT --Ns 8:160:4                       # passo 4
python main_simulation.py init --preset full --out $SKGAME_OUT --backends qpu:zephyr:clique        # solo Zephyr
python main_simulation.py init --preset full --out $SKGAME_OUT --chain-strength-prefactor 1.0 --chain-break-method minimize_energy
python main_simulation.py init --preset full --out $SKGAME_OUT --num-reads 2000 --annealing-time 50
```
Per scegliere le impostazioni delle catene, vedi `../visualization/embedding_explanation/chains_and_tuning.md` (§ 5, protocollo suggerito). Per fissare un QPU preciso: `python main_simulation.py qpu --out $SKGAME_OUT --backend qpu:zephyr:clique --solver Advantage2_system1.x`.

## 10. Risultati

```
$SKGAME_OUT/
  config.json                                  parametri (fissati da init)
  instances/<classe>/N0008.npz                 le matrici J (stesse per classico, gioco e QPU)
  classical/<classe>/N0008.json                riferimento + solver classici
  game/<regime>/N0008.json                     reward e fase del gioco
  embeddings/<solver>/clique/N0064.json        catene usate sul QPU, per solver
  anneal/qpu_zephyr_clique/<classe>/N…_i….json tutte le letture, timing, chain strength, chain break, letture "polished"
  anneal/qpu_pegasus_clique/...                idem su Pegasus
  summary/summary.csv                          una riga per (metodo, classe o regime, N)
  summary/backend_comparison.csv + .png        Zephyr vs Pegasus: p_success, residuo, chain break, lunghezza catene, vincitore
  summary/embeddings.csv                       qubit e lunghezza delle catene per back-end e N
```
`analyze` stampa anche il conteggio dei punti vinti da ciascun back-end. Le colonne `p_success_*` sono la frazione delle letture QPU che raggiunge l'ottimo; `p_success_polished` è la stessa frazione dopo una discesa greedy classica di ogni lettura.

Notebook: `sbatch -A $SKGAME_ACCOUNT cineca/job_notebook.sbatch` (richiede il wheelhouse con `WITH_NB=1`) produce `$SKGAME_OUT/main_simulation_executed.ipynb`. In alternativa, `bash cineca/sync.sh pull-results` e apri il notebook in locale con `SKGAME_OUT=results/full`.

## 11. Budget QPU (preset `full`, 2 back-end)

Chiamate per back-end:

$$n_{\rm call}=|N_s|\cdot n_{\rm classi}\cdot n_{\rm inst}=20\cdot4\cdot5=400$$

Tempo per chiamata (programmazione ≈ 15 ms, readout ≈ 100 µs, delay ≈ 20 µs, anneal 20 µs, 1000 letture):

$$t_{\rm call}\approx15\,{\rm ms}+1000\cdot(20+100+20)\,\mu{\rm s}=15\,{\rm ms}+1000\cdot140\,\mu{\rm s}=15\,{\rm ms}+140\,{\rm ms}=155\,{\rm ms}$$

Totale, due topologie:

$$T\approx2\cdot400\cdot0.155\,{\rm s}=2\cdot62\,{\rm s}=124\,{\rm s}\ \text{di QPU access time}$$

È una stima per eccesso sul readout. `qpu --check` dà quella del modello di timing di ogni solver, e il log riporta il valore reale chiamata per chiamata.

## 12. Tempo CPU della parte classica

Misurato: una run di ogni solver a N = 160 richiede ≈ 22 s (tabu da solo 10 s). Per classe: 20 N · 5 istanze · 5 run · ~15 s ≈ 2–2.5 h su un core. Per questo il job ha 6 h e le 4 classi girano in parallelo.

## 13. Se la macchina non è Leonardo

| | Leonardo | Galileo100 |
|---|---|---|
| login | `login.leonardo.cineca.it` | `login.g100.cineca.it` |
| partizione CPU (`SKGAME_PARTITION`) | `dcgp_usr_prod` (o `boost_usr_prod`) | `g100_usr_prod` |
| partizione seriale (`job_qpu`, `job_analyze`, `job_notebook`, `check_internet`) | `lrd_all_serial` | `g100_all_serial` |

Controlla sempre i nomi con `sinfo -s`.

## 14. Problemi frequenti

* **`config.json not found`**: manca `init` per quella cartella `--out` (o `sync.sh pull-inputs`).
* **`no embedding of K_N`** oppure N saltati: N oltre la clique del QPU reale, che ha qubit difettosi. `--check` stampa il massimo vero.
* **Nessun solver con `topology__type`**: il progetto Leap non ha accesso a un QPU di quella famiglia. Controlla `dwave solvers -la`, poi togli quel back-end con `init --backends ... --force` (le istanze restano).
* **`SolverAuthenticationError` / 401**: token sbagliato o di un altro progetto Leap.
* **Timeout di rete**: ogni chiamata viene ritentata 3 volte, poi lo stadio si ferma. Rilancia e riparte dalle chiamate mancanti.
* **`install_offline.sh` si ferma sulla versione di Python**: ricostruisci il wheelhouse con la versione stampata.
