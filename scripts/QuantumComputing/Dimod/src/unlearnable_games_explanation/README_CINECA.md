# Eseguire `main_simulation` su Cineca (Leonardo) con il QPU D-Wave

Questa guida spiega passo per passo come lanciare la campagna "SK-game vs quantum annealing" su Leonardo:
la parte classica sui nodi di calcolo, la parte quantistica sul QPU D-Wave in cloud (Leap).

Ipotesi fatte (cambiale se non valgono, vedi § 12):
* la macchina è **Leonardo**;
* il progetto Leap di Cineca è **già approvato** e hai il suo **token API** (se il ticket a superc@cineca.it è ancora aperto, puoi fare tutto tranne i passi 5 e 9: usa il simulatore `sim` al posto di `qpu`).

---

## 0. Cosa fa la simulazione

Per ogni **regime** e per ogni **N** in una griglia lineare (preset `full`: N = 8, 16, 24, …, 160), sulle **stesse** istanze J:

| regime | ε | J0 | α | β | problema del planner | fase del gioco | atteso |
|---|---|---|---|---|---|---|---|
| `ferro_easy` | 0 | 3 | 0.1 | ∞ | ferromagnete | punti fissi | ottimo facile per tutti |
| `sg_boundary` | 0 | 1 | 0.1 | ∞ | bordo SG/ferro | punti fissi | intermedio |
| `sg_satisficing` | 0 | 0 | 0.1 | ∞ | spin glass | punti fissi | ottimo difficile; il gioco si ferma sotto l'ottimo |
| `sg_cycles` | 0 | 0 | 1 | ∞ | spin glass | 2-cicli | il gioco non lo trova mai |
| `nonreciprocal_chaos` | 1 | 0 | 0.1 | ∞ | spin glass | caotica | il gioco non lo trova mai |
| `paramagnet` | 0 | 0 | 0.1 | 0.25 | spin glass | paramagnete (T=4 > T_c=2) | gioco casuale |

Perché questi punti: il problema di ottimizzazione $\max_S S^\top W S$ dipende solo da
$J_0^{\rm eff}=J_0/(1-\varepsilon/2)$ (ε da solo riscala W); per $J_0^{\rm eff}<1$ è uno spin glass (difficile),
per $J_0^{\rm eff}>1$ un ferromagnete (facile). La fase del **gioco** dipende invece da (α, β, ε).
I regimi con lo stesso (ε, J0) condividono le istanze: le **classi di problema** sono 4
(`eps0_J03`, `eps0_J01`, `eps0_J00`, `eps1_J00`), e l'annealer viene chiamato una volta per classe, non per regime.

Su ogni istanza si calcolano: ottimo di riferimento (esatto per N ≤ 24, migliore noto sopra), solver classici
(greedy, SA, PT, τ-EO, SDP, dimod SA/tabu), **annealer quantistico** (`DWaveCliqueSampler` via dimod) e il **gioco**.

La campagna è divisa in **stadi indipendenti e riprendibili** (un file già scritto non viene ricalcolato):

| stadio | comando | dove gira | internet? |
|---|---|---|---|
| init | `python main_simulation.py init --preset full --out DIR` | login | no |
| classico + gioco | `python main_simulation.py classical --out DIR --task i` | nodi di calcolo (array Slurm, i = 0..3) | no |
| QPU | `python main_simulation.py qpu --out DIR` | dove c'è HTTPS in uscita (`lrd_all_serial` o login) | **sì** |
| analisi | `python main_simulation.py analyze --out DIR` | ovunque | no |

Lo stadio QPU legge solo le istanze e chiama il sampler: quasi zero CPU, quindi può girare su un nodo di login.

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
cd Projects_master_HPQC
git checkout claude/sweet-rubin-wc3jb3     # finché il branch non è unito a main; dopo: git checkout main
```
Se il repository è privato, GitHub chiede username + **personal access token** (non la password).

Aggiornamenti successivi:
```bash
cd $WORK/Projects_master_HPQC && git pull
```

D'ora in poi lavora nella cartella del pacchetto:
```bash
cd $WORK/Projects_master_HPQC/scripts/QuantumComputing/Dimod/src/unlearnable_games_explanation
```

## 3. Ambiente Python (una volta, sul login: i nodi di calcolo non scaricano da internet)

```bash
module load python                          # serve Python >= 3.10; vedi `module avail python`
python -m venv $WORK/venvs/skgame
source $WORK/venvs/skgame/bin/activate
pip install --upgrade pip
pip install -e ".[dwave,qpu,test,nb]"       # numpy scipy numba dimod dwave-samplers dwave-system pytest jupyter
pytest -q                                   # 99 test, nessun token richiesto (~1 min)
```

## 4. Configura `cineca/env.sh` (una volta)

Apri `cineca/env.sh` e cambia le prime righe:
```bash
export SKGAME_ACCOUNT="${SKGAME_ACCOUNT:-CHANGE_ME}"          # -> il tuo account di `saldo -b`
export SKGAME_PARTITION="${SKGAME_PARTITION:-dcgp_usr_prod}"  # partizione CPU; se il budget è solo Booster: boost_usr_prod
export SKGAME_VENV="${SKGAME_VENV:-$WORK/venvs/skgame}"
export SKGAME_OUT="${SKGAME_OUT:-$WORK/skgame_results/full}"
```
Poi: `source cineca/env.sh` (carica modulo, venv e variabili; ogni job lo fa da solo).

## 5. Token Leap (una volta)

Il token si copia dalla dashboard Leap (https://cloud.dwavesys.com/leap/), **selezionando il progetto Cineca** (il token è per progetto).
```bash
dwave config create          # alla domanda "Authentication token" incolla il token; il resto: Invio
chmod 600 ~/.config/dwave/dwave.conf
dwave ping                   # risponde con il solver usato (spende pochi ms di QPU)
dwave solvers -la            # deve comparire almeno un Advantage* (QPU)
```
In alternativa, senza file: `export DWAVE_API_TOKEN=...` (mettilo in `~/.bashrc`, **mai** nel repository).
La `$HOME` è condivisa: anche i job vedono `~/.config/dwave/dwave.conf`.

## 6. Dove c'è internet?

```bash
bash cineca/check_internet.sh
```
Stampa un codice HTTP (raggiungibile) o `TIMEOUT`/`000` (no internet) per: login, `lrd_all_serial`, la tua partizione CPU.

| risultato | come lanciare lo stadio QPU |
|---|---|
| `lrd_all_serial` raggiungibile | `sbatch -A $SKGAME_ACCOUNT cineca/job_qpu.sbatch` (è già così in `submit_all.sh`) |
| solo login raggiungibile | `bash cineca/run_qpu_login.sh` (processo in background con `nohup`) |
| anche la partizione CPU | va bene tutto; puoi anche usare un job normale |

`lrd_all_serial` è la partizione seriale che gira sui nodi di login (max 4 h, 4 core, non scala il budget):
è la scelta migliore per il client QPU. Verifica i limiti con `sinfo -p lrd_all_serial`.

## 7. Prova a vuoto (2 minuti, nessun token, sul login)

```bash
source cineca/env.sh
python main_simulation.py init --preset smoke --out $WORK/skgame_results/smoke
python main_simulation.py classical --out $WORK/skgame_results/smoke     # ~20 s
python main_simulation.py sim       --out $WORK/skgame_results/smoke     # annealer simulato
python main_simulation.py analyze   --out $WORK/skgame_results/smoke
ls $WORK/skgame_results/smoke/summary/                                    # summary.csv + 4 figure .png
```

## 8. Prova sul QPU (pochi secondi di QPU)

```bash
python main_simulation.py qpu --out $WORK/skgame_results/smoke --check          # solver, clique max, stima tempo: NON spende QPU
python main_simulation.py qpu --out $WORK/skgame_results/smoke --max-calls 2    # 2 chiamate vere
python main_simulation.py analyze --out $WORK/skgame_results/smoke
```
Nel log ogni chiamata stampa `qpu_access=...ms (total ...s)`: è il tempo che Leap scala.
Il valore esatto per ogni chiamata è salvato in `anneal/qpu/<classe>/N…_i….json` → `info.timing_us.qpu_access_time` (µs).
Rilancia lo stesso comando senza `--max-calls` per completare la campagna smoke.

## 9. Campagna completa

```bash
source cineca/env.sh
python main_simulation.py init --preset full --out $SKGAME_OUT      # crea $SKGAME_OUT/config.json e stampa il piano
python main_simulation.py qpu --out $SKGAME_OUT --check             # stima del QPU dal modello del solver
bash cineca/submit_all.sh                                           # array classico (4 task) + job QPU + analisi in coda
```
`submit_all.sh` sottomette:
1. `job_classical.sbatch`: array `0-3`, una classe di problema per task, 6 h, sulla partizione `$SKGAME_PARTITION`;
2. `job_qpu.sbatch`: su `lrd_all_serial` (se al passo 6 non c'era internet lì: `bash cineca/submit_all.sh --no-qpu` e poi `bash cineca/run_qpu_login.sh`);
3. `job_analyze.sbatch`: parte quando 1 e 2 sono finiti (`afterany`).

Controllo:
```bash
squeue --me
tail -f logs/skgame-classical_*_0.out       # oppure logs/skgame-qpu_*.out, logs/qpu_login.log
```
Se un job finisce per walltime o cade la rete: **risottometti lo stesso comando**, riparte da dove era.
```bash
sbatch -A $SKGAME_ACCOUNT -p $SKGAME_PARTITION cineca/job_classical.sbatch   # solo i (classe, N) mancanti
sbatch -A $SKGAME_ACCOUNT cineca/job_qpu.sbatch                              # solo le chiamate mancanti
sbatch -A $SKGAME_ACCOUNT cineca/job_analyze.sbatch
```
L'analisi si può rilanciare quando vuoi, anche a metà campagna: usa ciò che c'è.

Varianti utili di `init` (prima di lanciare; `--force` per sovrascrivere un config esistente):
```bash
python main_simulation.py init --preset full --out $SKGAME_OUT --Ns 8:160:4          # passo 4 invece di 8
python main_simulation.py init --preset full --out $SKGAME_OUT --num-reads 2000 --annealing-time 50
python main_simulation.py init --preset full --out $SKGAME_OUT --regimes ferro_easy sg_satisficing
```
Con un sottoinsieme di regimi le classi possono essere meno di 4: `python main_simulation.py plan --out $SKGAME_OUT`
le elenca, e va adattato `#SBATCH --array=0-3` in `cineca/job_classical.sbatch`.
Le N oltre la clique massima del QPU (~170 su Advantage; `--check` la stampa) sono saltate automaticamente dallo stadio QPU.

## 10. Risultati

```
$SKGAME_OUT/
  config.json                         parametri della campagna (fissati da init)
  instances/<classe>/N0008.npz        le matrici J (la stessa istanza per classico, gioco e QPU)
  classical/<classe>/N0008.json       riferimento + reward di ogni solver classico per istanza e run
  game/<regime>/N0008.json            reward e fase (periodo) del gioco per istanza e condizione iniziale
  anneal/qpu/<classe>/N0008_i0.json   tutte le letture del QPU (reward distinti + conteggi), timing, chain break
  anneal/sim/...                      lo stesso con il simulatore
  summary/summary.csv                 una riga per (metodo, classe o regime, N)
  summary/*.png                       p_success e residuo vs N, TTS99 dell'annealer, fasi del gioco vs N
```
Colonne principali di `summary.csv`: `p_success_median` (QPU: frazione delle letture che raggiunge l'ottimo),
`residual_median` = (R_ref − R)/R_ref, `tts99`, `qpu_access_s`, `chain_break_fraction`,
`frac_fixed_point / frac_cycle / frac_no_recurrence` (gioco), `ref_from_annealer` (quante istanze in cui il QPU ha
migliorato il riferimento classico, possibile solo sopra N = 24).

**Notebook**: `main_simulation.ipynb` è un'interfaccia agli stessi stadi.
* headless su Cineca (solo analisi dei risultati già prodotti, salva il notebook eseguito con le figure):
  ```bash
  sbatch -A $SKGAME_ACCOUNT cineca/job_notebook.sbatch       # -> $SKGAME_OUT/main_simulation_executed.ipynb
  ```
  Le variabili `SKGAME_SAMPLER` (`none`/`sim`/`qpu`) e `SKGAME_RUN_CLASSICAL` (`0`/`1`) scelgono cosa eseguire.
* interattivo: copia la cartella dei risultati sul tuo computer e apri lì il notebook con `SKGAME_OUT` puntato alla copia.

**Riportare i risultati sul tuo computer** (dal tuo computer, non da Leonardo):
```bash
rsync -av <username>@login.leonardo.cineca.it:/leonardo_work/<ACCOUNT>/skgame_results/full/ ./results/full/ \
      --exclude instances/
```

## 11. Budget QPU (preset `full`)

Numero di chiamate (una chiamata per istanza, 4 classi di problema):

$$n_{\rm call}=|N_s|\cdot n_{\rm classi}\cdot n_{\rm inst}=20\cdot 4\cdot 5=400$$

Tempo per chiamata (valori tipici Advantage: programmazione ≈ 15 ms, readout ≈ 100 µs, delay ≈ 20 µs; anneal 20 µs, 1000 letture):

$$t_{\rm call}\approx t_{\rm prog}+n_{\rm reads}\,(t_a+t_{\rm ro}+t_d)=15\,{\rm ms}+1000\cdot(20+100+20)\,\mu{\rm s}=15\,{\rm ms}+1000\cdot140\,\mu{\rm s}=15\,{\rm ms}+140\,{\rm ms}=155\,{\rm ms}$$

Totale:

$$T\approx 400\cdot 0.155\,{\rm s}=62\,{\rm s}\ \text{di QPU access time}$$

È una stima per eccesso sul readout; `qpu --check` dà quella del modello di timing del solver e il log riporta il valore reale.
Con `--Ns 8:160:4` le N diventano 39 e quindi $n_{\rm call}=39\cdot4\cdot5=780$, $T\approx780\cdot0.155\approx121$ s.

Tempo CPU della parte classica (misurato, una run di ogni solver a N = 160 ≈ 22 s, tabu da solo 10 s):
per classe ≈ 20 N · 5 istanze · 5 run · ~15 s ≈ 2–2.5 h su un core, per questo il job ha 6 h e le 4 classi girano in parallelo.

## 12. Se la macchina non è Leonardo

| | Leonardo | Galileo100 |
|---|---|---|
| login | `login.leonardo.cineca.it` | `login.g100.cineca.it` |
| partizione CPU (`SKGAME_PARTITION`) | `dcgp_usr_prod` (o `boost_usr_prod`) | `g100_usr_prod` |
| partizione seriale (in `job_qpu.sbatch`, `job_analyze.sbatch`, `job_notebook.sbatch`, `check_internet.sh`) | `lrd_all_serial` | `g100_all_serial` |

Il resto è identico. Controlla sempre i nomi con `sinfo -s`.

## 13. Problemi frequenti

* **`config.json not found`**: manca `init` per quella cartella `--out`.
* **`Cannot embed given BQM`**: N oltre la clique del QPU; lo stadio QPU le salta, se succede verifica il solver con `--check` o scegli un QPU con `--solver Advantage_system4.1`.
* **`SolverAuthenticationError` / 401**: token sbagliato o di un altro progetto Leap (§ 5).
* **Timeout di rete nel job QPU**: ogni chiamata viene ritentata 3 volte, poi il job si ferma; risottometti, riparte dalle chiamate mancanti.
* **Processo ucciso sul login**: rilancia `bash cineca/run_qpu_login.sh`; le chiamate finite restano.
* **`module load python` non trova la versione**: `module avail python` e carica esplicitamente una ≥ 3.10 (anche in `cineca/env.sh`).
