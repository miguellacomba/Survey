#Streamlit web‑app for eliciting healthcare utility preferences by means of two 
#methods: Standard‑Gamble (SG), and Pairwise Comparison (PC).

#A page index drives the navigation flow:

#    0 → survey setup
#    1 → device availability
#    2 → respondent intro
#    6 → Pairwise Comparison
#    5 → Standard Gamble
#  120 → Thank-you
#   98 → optimisation-setup (solo cuando meta["finished"])
#   99 → analytics

#The code is organised in self‑contained view functions, each responsible for 
#rendering one logical page and mutating the session state so that the main
#"main()" dispatcher can pick the next view.

import streamlit as st
import json
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from scipy.optimize import linprog
import matplotlib.pyplot as plt 
from itertools import combinations
from adjustText import adjust_text

################################################################################
#  Global constants                                                            #
################################################################################

#ALTERNATIVES = ["Critical", "Emergency", "Scheduled", "Momentary"]

dev_load_map = [
    "Cocina eléctrica",
    "Refrigerador solar para vacunas",
    "Ecógrafo",
    "Concentrador de oxígeno",
    "Ollas eléctricas",
    "Lámpara de cuello de cisne",
#   "Proyector de vídeo",
#   "Electrocardiógrafo",
#   "Nebulizador",
#   "Unidad dental",
#   "Refrigerador",
#   "Camilla eléctrica",
#   "Aspirador de secreciones",
#   "Esterilizador",
#   "Bomba de infusión",
#   "Monitor de signos vitales",
#   "Compresor de aire",
#   "Ordenador de sobremesa",
#   "Portátil",
#   "Conexión a Internet",
#   "Impresora",
#   "Aire acondicionado",
#   "Bombillas",
]

power_map = {
    "Cocina eléctrica":                 8000,
    "Refrigerador solar para vacunas":  1104,
    "Ecógrafo":                         1158,
    "Concentrador de oxígeno":          1180,
    "Ollas eléctricas":                 4000,
    "Lámpara de cuello de cisne":        672,
#   "Proyector de vídeo":                 32.5,
#   "Electrocardiógrafo":                 17.5,
#   "Nebulizador":                       120,
#   "Unidad dental":                    7200,
#   "Refrigerador":                      576,
#   "Camilla eléctrica":                 720,
#   "Aspirador de secreciones":          220,
#   "Esterilizador":                    6000,
#   "Bomba de infusión":                 200,
#   "Monitor de signos vitales":        1200,
#   "Compresor de aire":                5840,
#   "Ordenador de sobremesa":            720,
#   "Portátil":                          180,
#   "Conexión a Internet":               288,
#   "Impresora":                        1200,
#   "Aire acondicionado":               2700,
#   "Bombillas":                        7920,
}

################################################################################
#  Device “buckets”                                                            #
################################################################################

DEVICE_GROUPS: dict[str, list[str]] = {
    "Diagnóstico / monitorización clínica": [
        "Ecógrafo",
#       "Electrocardiógrafo",
#       "Monitor de signos vitales",
    ],
    "Tratamiento clínico / soporte vital": [
        "Concentrador de oxígeno",
#       "Nebulizador",
#       "Bomba de infusión",
#       "Aspirador de secreciones",
#       "Camilla eléctrica",
    ],
#   "Odontología": [
#       "Unidad dental",
#       "Compresor de aire",
#   ],
    "Cadena de frío / esterilización": [
        "Refrigerador solar para vacunas",
#       "Refrigerador",
#       "Esterilizador",
    ],
    "Cocina / nutrición": [
        "Cocina eléctrica",
        "Ollas eléctricas",
    ],
    "Iluminación y pequeños equipos": [
        "Lámpara de cuello de cisne",
#       "Bombillas",
    ],
#   "Oficina / informática y formación": [
#       "Ordenador de sobremesa",
#       "Portátil",
#       "Conexión a Internet",
#       "Impresora",
#       "Proyector de vídeo",
#   ],
#   "Climatización / servicios edificio": [
#       "Aire acondicionado",
#   ],
}

################################################################################
#  Session‑state bootstrap                                                     #
################################################################################
    #The following keys are created once per browser session.  Every view checks
    #or mutates these keys but never *deletes* them (except for page‑local temp
    #keys which are removed explicitly)."""
################################################################################

if "page_index" not in st.session_state:              # global navigation pointer
    st.session_state.page_index = 0
    
if 'page_index_sg' not in st.session_state:           #To move inside the SG method
    st.session_state.page_index_sg = 0
    
if 'responses_sg' not in st.session_state:            #Store the answers of the SG method
    st.session_state.responses_sg = {}

if "responses_pc" not in st.session_state:            #To store the answers of the PC method  
    st.session_state.responses_pc = {}

if 'page_index_pc' not in st.session_state:           #To move inside the PC method
    st.session_state.page_index_pc = 0

#We initialize the wins dictionary. This dict will show all the devices (dev) wins.
if "wins_pc" not in st.session_state or set(st.session_state["wins_pc"].keys()) != set(dev_load_map):
    st.session_state["wins_pc"] = {d: set() for d in dev_load_map}

if 'checked_pairs_pc' not in st.session_state:        #Checked pairs, to save if that pair has been checked already or not
    st.session_state['checked_pairs_pc']= set()

if "num_respondents" not in st.session_state:         #We'll store the total number of respondents
    st.session_state.num_respondents = None
    
if "current_respondent_num" not in st.session_state:  #We'll track which respondent we're on (1..N).
    st.session_state.current_respondent_num = 1
    
#We'll store the results for each respondent in a dictionary
if "survey_data" not in st.session_state or not isinstance(st.session_state.survey_data, list):
    st.session_state.survey_data = []
    
if "this_respondent_id" not in st.session_state:     # ID currently entering answers
    st.session_state.this_respondent_id = None

if "ids" not in st.session_state:                    # list[str] of respondent IDs
    st.session_state.ids = []

if "facility_devices" not in st.session_state:       # devices available in the facility
    st.session_state.facility_devices = set()

if "selected_method" not in st.session_state:        #"SG"/"PC"/"ES"
    st.session_state.selected_method = None          

#if "current_idx" not in st.session_state:            #0-based respondent index
#    st.session_state.current_idx = 0                 

if "assignments" not in st.session_state:            # maps each device to a chosen load type (None = not chosen yet)
    st.session_state.assignments = {d: None for d in dev_load_map}

if "max_power" not in st.session_state:          # W – capacity of the system
    st.session_state.max_power = None
    
if "utility_source" not in st.session_state:     # "PC", "SG", or "Average"
    st.session_state.utility_source = None

if "ids" not in st.session_state:
    st.session_state.ids = []            # will grow with .append()

######################################################################
#  Global look & feel – bump all fonts up
######################################################################
def set_global_font(base_px: int = 18) -> None:
    """Inject a <style> block that enlarges everything.

    `base_px` becomes the default size for normal text.
    Headings & widget labels get scaled proportionally.
    """
    st.markdown(
        f"""
        <style>
            html, body, [class*="css"]  {{
                font-size: {base_px}px !important;
            }}
            h1 {{ font-size: {base_px*1.8}px !important; }}
            h2 {{ font-size: {base_px*1.55}px !important; }}
            h3 {{ font-size: {base_px*1.35}px !important; }}

            /* make widget-labels a bit larger too */
            label, .stRadio > label, .stCheckbox > label,
            .stTextInput > label, .stSlider > label {{
                font-size: {base_px*1.05}px !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

set_global_font(30)

################################################################################
#  Persistent storage                                                          #
################################################################################

SCRIPT_DIR = Path(__file__).resolve().parent          # one folder for the whole study
DATA_DIR   = SCRIPT_DIR / "survey_data"  
RESP_PATTERN  = "respondent_{rid}.json"       # one file per respondent
META_FILE     = DATA_DIR / "survey_meta.json" # holds target_n & finished flag

DATA_DIR.mkdir(exist_ok=True)

st.write(f"Using DATA_DIR = {DATA_DIR}")

def load_meta():
    if META_FILE.exists():
        return json.loads(META_FILE.read_text())
    return None

def save_meta(meta):
    META_FILE.write_text(json.dumps(meta, indent=2))

def load_all_responses():
    """
    Return a list of respondent dicts.
    Silently skips files that are not valid JSON or have no 'id'.
    """
    records = []
    for p in DATA_DIR.glob("respondent_*.json"):
        st.write("📄 scanning", p.name)
        try:
            rec = json.loads(p.read_text())
        except json.JSONDecodeError as err:
            st.warning(f"⚠️  {p.name} invalid JSON ({err}) – skipped.")
            continue
        if not isinstance(rec, dict) or "id" not in rec:
            st.warning(f"⚠️  {p.name} has no 'id' key – skipped.")
            continue
        records.append(rec)
    st.write(f"✅ loaded {len(records)} respondent file(s) from disk")
    return records

if "survey_meta" not in st.session_state:
    st.session_state.survey_meta  = load_meta() or {}          # may be empty

meta = st.session_state.survey_meta

##############################################################################
#  Enrutado automático al arrancar                                           #
##############################################################################

# max_power
if meta.get("max_power") is not None:
    st.session_state.max_power = meta["max_power"]

if "utility_source" in meta:
    st.session_state.utility_source = meta["utility_source"]

if not st.session_state.facility_devices:
    st.session_state.facility_devices = set(meta.get("facility_devices", []))     # only if still empty 

# facility_devices
if not st.session_state.facility_devices:        
    st.session_state.facility_devices = set(meta.get("facility_devices", []))     # only if still empty
    
if "survey_data" not in st.session_state:
    st.session_state.survey_data  = load_all_responses()       # list[dict]

st.session_state.completed_ids = {rec["id"] for rec in st.session_state.survey_data}

st.write(f"🔍 Loaded respondents on disk: {st.session_state.completed_ids}")
    
if "completed_ids" not in st.session_state:
    st.session_state.completed_ids = { rec["id"] for rec in st.session_state.survey_data }

#############################################################################
# Password                                                                  #
#############################################################################

PASSWORD = st.secrets.get("APP_PASSWORD") or os.getenv("APP_PASSWORD")

if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False

if not st.session_state.auth_ok:
    st.title("🔒 Acceso restringido")
    pwd = st.text_input("Introduce la contraseña:", type="password")
    if st.button("Entrar"):
        if pwd == PASSWORD:
            st.session_state.auth_ok = True
            st.rerun()
        else:
            st.error("❌ Contraseña incorrecta")
    st.stop()

################################################################################
#  Helper utilities                                                            #
################################################################################

def normalise_answer(method_code, answer, n=len(dev_load_map)):
    #Here, we will normalize the utilities between the different methods.

    #SG already delivers utilities in percent, so we simply copy them. 
    #PC gives the preferences in order; we map the first position to 100, the last to 0 via 
    #a linear scale.

    """
    • SG answers are already percentages → return unchanged.
    • PC rankings are mapped linearly so that
        rank #1 → 100.0
        rank #n →   0.1      (instead of 0.0)
    """
    if method_code == "SG":
        return answer

    floor = 0.1                 # utility for the last-ranked device
    span  = 100.0 - floor       # 99.9 to distribute linearly

    util = {}
    for rank, dev in enumerate(answer, start=1):     # 1-based rank
        util[dev] = ((n - rank) / (n - 1)) * span + floor

    # return sorted high→low (optional; handy elsewhere)
    return dict(sorted(util.items(), key=lambda kv: -kv[1]))

def filter_and_rescale_for_optim(util_series, avail_set, renorm=True):
    #keep only devices available in the facility. If renorm is True, linearly 
    #rescale so that max ⇒ 1.0 and min ⇒ 0.0 (unless all utilities are equal, 
    #in which case everything becomes 1.0). Returns a new Series.

    sub = util_series[util_series.index.isin(avail_set)].copy()
    if not renorm or sub.empty:
        return sub

    umin, umax = sub.min(), sub.max()
    if umax == umin:
        sub[:] = 1.0                          # avoid divide-by-zero
    else:
        sub = (sub - umin) / (umax - umin)    # 0-to-1

    return sub
    
################################################################################
#  View functions – one per *page_index*                                       #
################################################################################

def survey_setup_page():
    # refrescar contadores cada vez que se abre el menú de organización
    st.session_state.survey_data = load_all_responses()
    st.session_state.completed_ids = {
        r["id"] for r in st.session_state.survey_data
    }

    meta  = st.session_state.survey_meta
    done  = len(st.session_state.completed_ids)
    plural = "" if done == 1 else "s"

    st.title("Configuración de la encuesta — solo para la persona organizadora")

    # ── 1. Preguntar tamaño muestral si aún no existe ──────────────────────
    if "target_n" not in meta:
        st.info(
            f"**Ya tienes** {done} participante{plural} guardado{plural} "
            "pero aún **no has fijado** el tamaño muestral objetivo.  \n"
            "Introduce el **número total** de participantes que piensas reclutar."
        )

        target = st.number_input(
            "Número total de participantes a reclutar:",
            min_value=max(1, done),
            step=1,
            value=max(2, done),
        )

        if st.button("Crear / actualizar encuesta"):
            meta.update({
                "target_n": int(target),
                "created": meta.get("created", datetime.utcnow().isoformat()),
                "finished": False,
            })
            save_meta(meta)
            st.success("Objetivo guardado — ya puedes continuar recopilando datos.")
            st.rerun()
        return

    # ── 2. Marcar encuesta como finalizada si se alcanza la cuota ───────────
    if not meta.get("finished") and done >= meta["target_n"]:
        meta["finished"] = True
        save_meta(meta)
        st.session_state.page_index = 98   # salto directo a optimización
        st.rerun()
        return

    # ── 3. Mostrar progreso & navegación ───────────────────────────────────
    st.info(
        f"Tamaño muestral objetivo: **{meta.get('target_n', '…')}**  |  "
        f"Completado: **{done}**"
    )

    if meta.get("finished"):
        st.success("¡Todos los participantes han terminado — análisis disponible!")
        if st.button("Ir al análisis"):
            st.session_state.page_index = 99
            st.rerun()
    else:
        # Si los dispositivos ya están definidos, saltamos a la intro del encuestado
        next_page = 2 if "facility_devices" in meta else 1
        if st.button("Seguir recopilando datos"):
            st.session_state.page_index = next_page
            st.rerun()

def device_availability_page():
    st.title("Dispositivos disponibles – organiser only")
    st.write("Ticka los dispositivos **disponibles** en su Centro de salud:")

    # --- show buckets -------------------------------------------------------
    for group_name, dev_list in DEVICE_GROUPS.items():
        st.subheader(group_name)

        # one checkbox per device (spread across 3 columns for compactness)
        n_cols = min(3, len(dev_list))
        cols   = st.columns(n_cols)

        for i, dev in enumerate(dev_list):
            col = cols[i % n_cols]
            with col:
                checked = st.checkbox(
                    dev,
                    key=f"chk_{dev}",
                    value=(dev in st.session_state.facility_devices),
                )
            if checked:
                st.session_state.facility_devices.add(dev)
            else:
                st.session_state.facility_devices.discard(dev)

    # --- save & move on -----------------------------------------------------
    if st.button("Confirm devices"):
        meta = st.session_state.survey_meta
        meta["facility_devices"] = sorted(st.session_state.facility_devices)
        save_meta(meta)

        st.success("Saved. You won’t be asked again.")
        st.session_state.page_index = 2         # jump to respondent intro
        st.rerun()

def optimisation_setup_page():
    st.title("Optimisation setup – for organiser only")

    meta = st.session_state.survey_meta

    # ------ power capacity ---------------------------------------------------
    pow_val = st.number_input(
        "Power available for critical loads (W):",
        min_value=100,
        step=100,
        value=int(meta.get("max_power", 10_000)),
    )

    # ------ which utilities to use -------------------------------------------
    util_val = st.radio(
        "Utilities to use for optimisation:",
        ["PC", "SG", "Average"],
        index=["PC", "SG", "Average"].index(meta.get("utility_source", "Average")),
    )

    # ------ save & return ----------------------------------------------------
    if st.button("Save and return"):
        meta["max_power"]     = int(pow_val)
        meta["utility_source"] = util_val
        save_meta(meta)

        # mirror into session-state for immediate use
        st.session_state.max_power      = int(pow_val)
        st.session_state.utility_source = util_val

        st.success("Settings saved.")
        st.session_state.page_index = 99
        st.rerun()

def configure_load_contents():
    st.header("Survey-taker only – Assign devices to each load")

    for device in dev_load_map:
    #A radio button forces a single choice
        choice = st.radio(
            label=device,
            options=ALTERNATIVES,
            horizontal=True,
            key=f"radio_{device}",
            index=ALTERNATIVES.index(st.session_state.assignments[device])
            if st.session_state.assignments[device] is not None
            else 0,
        )
        st.session_state.assignments[device] = choice
        st.divider()
        # Which devices are still unassigned?
        unassigned = [d for d, load in st.session_state.assignments.items() if load is None]
    
        #We disable the confirm button until everything is assigned
        confirm_disabled = len(unassigned) > 0
        
        if confirm_disabled:
            st.warning(
                f"Please assign a load type to every device. "
                f"Unassigned: {', '.join(unassigned)}"
            )

    if st.button("Confirm Load Type Devices", disabled=confirm_disabled):
        st.success("All devices assigned – thanks!")
        st.json(st.session_state.assignments)
        st.session_state.page_index = 4 

################################################################################
#  Respondent‑level pages                                                      #
################################################################################

def next_auto_id():
    """
    Devuelve el próximo ID disponible con prefijo 'SP' (SP1, SP2, …).
    Busca en los ficheros de respuestas ya guardados y en la sesión actual.
    """
    taken = {rec["id"] for rec in st.session_state.survey_data}              # en disco
    taken.update(st.session_state.ids)                                       # en memoria
    i = 1
    while f"SP{i}" in taken:
        i += 1
    return f"SP{i}"

def respondent_intro_page():

#    st.write("Before introducing your ID and proceeding to the method, it is important that you read and understand the following information regarding the management of your data.")
#    st.write("Place and time:                                             - To be introduced")
#    st.write("Contact person if you have any questions:                   - Miguel Lacomba Albert, ETH Zürich Student, mlacomba@student.ethz.ch")
#    st.write("Data Protection Officer ETH Zurich:		                  - Tomislav Mitar (tomislav.mitar@sl.ethz.ch)")
#    st.write("----------------------------------------------------------------------------")
#    st.write("We would like to ask you if you are willing to participate in our research project. Your participation is voluntary. Please read the text below carefully and ask the conducting person about anything you do not understand or would like to know.")
#    st.write("**What is investigated and how?**")
#    st.write("     This study investigates your preferences regarding powered medical devices and appliances through two structured in-person surveys. You will be presented with a hypothetical scenario involving limited electricity access and will respond to a series of questions using two standard preference elicitation techniques: Pairwise Comparison and Standard Gamble. Your responses will be used to derive utility values—representing the relative importance of each device—which will then serve as mathematical parameters in an optimization model. The goal is to identify the most critical electrical loads in healthcare settings based on these preferences.")
#    st.write("**Who can participate?**")
#    st.write("     To be eligible for participation, you must currently be employed in a healthcare facility and have familiarity with its day-to-day clinical operations. Participants must also be able to complete a 30-minute survey session on a computer.")
#    st.write("**What am I supposed to do as a participant?**")
#    st.write("     You will be asked to evaluate medical devices under different electricity availability scenarios and indicate your preferences. This involves responding to survey questions designed to capture your judgment about the criticality of each device.")
#    st.write("**What are my rights during participation?**")
#    st.write("     Participation in this study is entirely voluntary. You have the right to withdraw at any time without providing a reason and without any negative consequences.")
#    st.write("**What risks and benefits can I expect?**")
#    st.write("     There are no anticipated physical or psychological risks other than perhaps mild discomfort due to computer screen exposure and computer use. The study involves only answering two surveys, guided by the survey taker, and poses minimal inconvenience. Your participation contributes to research that may enhance the resilience of healthcare infrastructure in low-resource settings.")
#    st.write("**Will I be compensated for my participation?**")
#    st.write("     No financial or material compensation is provided for participation in this study.")
#    st.write("**What data is collected from me and how is it used?**")
#    st.write("     No personal identifying information will be collected. The study only records your survey responses, which are used to calculate utility scores for each device. These utilities help inform the optimization of critical load sets in healthcare facilities.")
#    st.write("----------------------------------------------------------------------------")

#    st.write("If you understand and agree with the mentioned above, please introduce the ID you will want to use and sign the consent form you were given in paper.")
    
    """
    Pantalla de bienvenida del participante.
    Muestra su identificador asignado automáticamente y un botón para comenzar.
    NO pide nada al usuario.
    """
    # ── si todavía no se ha generado un ID para esta sesión ─────────────────
    if st.session_state.this_respondent_id is None:
        new_id = next_auto_id()
        st.session_state.this_respondent_id = new_id
        st.session_state.ids.append(new_id)          # guardamos en sesión

    rid = st.session_state.this_respondent_id

    # ── UI ──────────────────────────────────────────────────────────────────
    st.title("Bienvenido/a a la encuesta")
    
    st.markdown(
        f"Tu identificador de participante es **{rid}**  \n"
        "Por favor, indícalo al investigador si se te solicita."
    )
    st.markdown("---")

    st.write("Hola, mi nombre es Miguel Lacomba Albert, y soy estudiante en la ETH de Zürich.")
    
    st.write("Actualmente estoy terminando mi Trabajo de Fin de Máster que trata la **priorización de cargas criticas en Centros de Salud**, cuando estos se encuentran en **situaciones de escasez de energía (apagones, acceso a la red inestable, etc)**.")

    st.write("[Consideramos una carga crítica, en el ámbito de los centros de salud, a aquella que si o si debe tener un acceso a la energía fiable y continuado, sin interrupciones].")

    st.write("Este Proyecto busca entonces ayudar a los centros de salud de regiones donde el acceso a la energía eléctrica no está asegurado. En este caso en concreto, está diseñado para un centro de salud en un poblado indígena cerca de Barranquilla, Colombia.")

    st.write("**¿Cómo?** Se ha diseñado una herramienta para ayudar al sistema a distribuir la energía hacia los aparatos considerados como más importantes en situaciones donde la energía escasea.")

    st.write("**¿No se ha hecho antes?** La respuesta es que sí, sí que se han estudios previos para establecer cuales son las cargas críticas en centros de salud, PERO, nunca considerando la opinión de doctores, enfermeras, y cualquier otro trabajador de un centro de salud con conocimientos médicos; sino que siempre ha sido considerando la opinión de los pacientes para ello. ")

    st.write("**¿Por qué necesitamos profesionales de la salud de España?** Si bien es cierto que en España o en cualquier país desarrollado, esto no es tan necesario, la opinión de los profesionales españoles (occidentales en nuestro caso), nos sirve para comprobar la consistencia y fiabilidad entre los métodos que estamos estudiando.")

    st.write("**¿Por qué necesitamos profesionales de la salud de España?** Si bien es cierto que en España o en cualquier país desarrollado, esto no es tan necesario, la opinión de los profesionales españoles (occidentales en nuestro caso), nos sirve para comprobar la consistencia y fiabilidad entre los métodos que estamos estudiando.")

    st.write("Entonces, a modo de resumen :")
    st.write("* **¿Quién puede participar?** Cualquier profesional de la salud.")
    st.write("- **¿Tiempo necesario?** Alrededor de **15 minutos**")
    st.write("- **¿Es anónimo?** Si, es **100% anónimo**")
    st.write("- **¿Hay compensación económica?** **No**, no hay compensación económica.")
    st.write("- **¿Cual es su Rol?** Su rol será simplemente el de contestar una encuesta que involucra dos métodos para evaluar cómo usted prioriza los aparatos médicos durante apagones, cortes de luz, etc.")

    st.markdown(
        "Cuando esté listo/a, pulsa el botón para comenzar."
    )

    if st.button("Comenzar encuesta"):
        # empezamos por PC
        st.session_state.page_index    = 6
        st.rerun()

###################################### Standard Gamble ##############################################

def standard_gamble_method():
    page_sg       = st.session_state.page_index_sg
    total_devices = len(dev_load_map)

    # ───────────────────────── Página de introducción ───────────────────────
    def sg_intro_page():
        st.title("Standard Gamble – introducción rápida")

        st.markdown(
            """
            En esta encuesta realizará un **_Standard Gamble (SG)_**
            para indicar qué dispositivos deberían recibir electricidad
            cuando esta escasea.

            ### 🌩️ Contexto
            * A veces la demanda **supera** lo que el centro puede generar.
            * Debe decidir cómo asignar esa energía limitada.

            ### 🎲 Sus tres opciones en cada paso son las siguientes:
            1. **<span style='color:#3CA4FF;'>Opción A – Energía parcial</span>**  
               Una pequeña cantidad de energía *garantizada*, pero sabemos que **dejará de funcionar** en cuanto se consuma.
            2. **<span style='color:#FF5733;'>Opción B – Lotería</span>**  
               En este caso, usted decide **Apostar**: Escoger un *p* % de probabilidad de que el dispositivo va a **funcionar sin problemas** y sin cortes,
               y por tanto, otorgar un *(1-p)* % de probabilidad a que **no va a funcionar en ningún momento**.
            3. **<span style='color:#AAAAAA;'>Indiferente</span>**  
               Usted es **indiferente**. Es decir, la probabilidad **P**% le parece un valor apropiado de **importancia** para este dispositivo. 

            ### ¿Qué tiene que hacer entonces usted? 
            * Primero, ver de qué dispositivo se trata.
            * Segundo, pensar qué tan importante es para usted. 
            * Tercero, **si** cree que la probabilidad inicial asignada (50% de que funcionará sin problemas- 50% de que no funcionará en ningún caso) **representa lo valioso que es para usted** este aparato, puede clicar en indiferente y avanzar al siguiente. Si, por el contrario, considera que es más importante para usted que funcione fiablemente el aparato en cuestión, debe seleccionar la opción A (para incrementar la probabilidad de que debe funcionar en cualquier caso) tantas veces como crea necesario hasta obtener la probabilidad **P**% deseada. En el caso de que crea que es **menos** importante en su opinión que la probabilidad inicial (es decir, para usted es un aparato más prescindible que otros), debe seleccionar tantas veces como considere la Opción B, hasta que la probabilidad represente la importancia que le asigna usted a este aparato. 
            * Cuarto, cuando la **probabilidad P% represente la importancia que usted le asigna al aparato, debe clicar en Indiferente para avanzar al siguiente aparato**.

            A continuación veremos una pantalla de Ejemplo, antes de avanzar al primer aparato.
            
            """, 
            unsafe_allow_html=True,
        )

        st.markdown("**Cuando esté listo/a, haga clic en el botón para empezar.**")

        if st.button("Ver ejemplo"):
            st.session_state.page_index_sg = -1   # ← demo page
            st.rerun()
            
#        if st.button("Comenzar SG"):
#            st.session_state.page_index_sg = 1
#            st.rerun()

# ------------------------------------------ Example -------------------------------------------

    def sg_example_page():
        """Pantalla interactiva de ejemplo (no se registra la respuesta)."""
        demo_dev = "Dispositivo de ejemplo"
        st.title("Ejemplo de pregunta")
        st.markdown(
            f"Imagine que el **{demo_dev}** puede recibir energía de forma poco fiable."
        )
        # Reutilizamos la misma UI que en las preguntas reales:
        dummy_res = sg_interactive_core(demo_dev, store_answer=False)
        # Botón para continuar
        if st.button("¡Entendido, empecemos!"):
            st.session_state.page_index_sg = 1    # primer dispositivo real
            st.rerun()

    def sg_interactive_core(device_name: str, store_answer: bool) -> None:
        """Construye la UI SG; si *store_answer* es False no guarda nada."""
        rid = st.session_state.this_respondent_id
        # inicializamos los pivotes binarios locales (no pasa nada si se recrean)
        k_min, k_max, k_guess = (f"{device_name}_{s}" for s in ("p_min", "p_max", "p_guess"))
        for k, v in [(k_min, 0.0), (k_max, 1.0), (k_guess, 0.5)]:
            st.session_state.setdefault(k, v)

        p_min   = st.session_state[k_min]
        p_max   = st.session_state[k_max]
        p_guess = st.session_state[k_guess]

        colA, colB, colC = st.columns([1.7, 1.7, 1.6], gap="small")
        st.markdown(
            """
            <style>
            div.bttn > button{width:100%;font-size:1.1rem;padding:1rem 0;
                              border-radius:10px;font-weight:600;}
            div.bttn.optA>button{background:#4CAF50;color:#fff;}
            div.bttn.optB>button{background:#2196F3;color:#fff;}
            div.bttn.optC>button{background:#9E9E9E;color:#fff;}
            </style>""",
            unsafe_allow_html=True,
        )

        choice_clicked = None
        with colA:
            st.markdown("### <span style='color:#3CA4FF;'>Opción A</span>", unsafe_allow_html=True)
            if st.container().button("Potencia parcial", key=f"A_{device_name}"):
                choice_clicked = "Partial"
        with colB:
            st.markdown("### <span style='color:#FF5733;'>Opción B</span>", unsafe_allow_html=True)
            st.markdown(f"Apuesta: **{p_guess*100:.0f}%** éxito, **{(1-p_guess)*100:.0f}%** fallo")
            if st.container().button("Apostar", key=f"B_{device_name}"):
                choice_clicked = "Lottery"
        with colC:
            st.markdown("### Indiferente", unsafe_allow_html=True)
            if st.container().button("Me da igual", key=f"C_{device_name}"):
                choice_clicked = "Indifferent"

        # ── manejo de clic ─────────────────────────────────────
        if choice_clicked is None:
            return                              # nada pulsado

        # ── clicks en A / B → ajustan rangos incluso en la demo
        if choice_clicked == "Partial":
            st.session_state[k_min] = p_guess
        elif choice_clicked == "Lottery":
            st.session_state[k_max] = p_guess
        elif choice_clicked == "Indifferent" and store_answer:
            # solo guardamos respuesta real, nunca en el ejemplo
            rid = st.session_state.this_respondent_id
            st.session_state.responses_sg.setdefault(rid, {})[device_name] = p_guess * 100
            st.session_state.page_index_sg += 1   # siguiente dispositivo
            st.rerun()
            return

        # recalcular nuevo punto medio y recargar la página
        st.session_state[k_guess] = (st.session_state[k_min] + st.session_state[k_max]) / 2
        st.rerun()
        
# ----------------------------------- Device Page (one per page) ---------------------------------
    
    def sg_interactive(index: int) -> None:
        """Muestra una pregunta SG con 3 botones grandes."""
        rid          = st.session_state.this_respondent_id
        device_name  = dev_load_map[index - 1]
        total_devs   = len(dev_load_map)

        # Encabezado ----------------------------------------------------------
        st.title("Standard Gamble – elige tu opción")
        st.subheader(f"Participante **{rid}**")
        st.markdown(f"**Dispositivo {index} de {total_devs}**")

        # Indicador de disponibilidad (emoji) ---------------------------------
        in_facility = device_name in st.session_state.facility_devices
        badge       = "🟢" if in_facility else "🟡"
        st.markdown(f"### {badge}  **{device_name}**")

        st.write(
            "Suponga que el dispositivo puede alimentarse pero es "
            "**POCO FIABLE**: puede apagarse por energía insuficiente o por cortes de luz, por ejemplo.")

        # Estado interno SG por dispositivo -----------------------------------
        k_min, k_max, k_guess = (f"{device_name}_{s}"
                                 for s in ("p_min", "p_max", "p_guess"))
        for k, v in [(k_min, 0.0), (k_max, 1.0), (k_guess, 0.5)]:
            st.session_state.setdefault(k, v)

        p_min   = st.session_state[k_min]
        p_max   = st.session_state[k_max]
        p_guess = st.session_state[k_guess]

        # Distribución de columnas --------------------------------------------
        colA, colB, colC = st.columns([1.7, 1.7, 1.6], gap="small")
    
        # tiny CSS tweak to keep the wide buttons looking good
        st.markdown(
            """
            <style>
            div.bttn > button{
                width:100%;font-size:1.15rem;padding:1.1rem 0.5rem;
                border-radius:10px;font-weight:600;
            }
            div.bttn.optA>button{background:#4CAF50;color:#fff;}
            div.bttn.optB>button{background:#2196F3;color:#fff;}
            div.bttn.optC>button{background:#9E9E9E;color:#fff;}
            </style>
            """,
            unsafe_allow_html=True,
        )
    
        choice_clicked = None
    
        with colA:
            st.markdown("### <span style='color:#3CA4FF;'>Opción&nbsp;A</span>",
                        unsafe_allow_html=True)
            st.markdown(
                f"El **{device_name}** funciona **A VECES**: por ejemplo, "
                "funciona la primera vez que lo necesita pero falla la siguiente"
            )
            if st.container().button("Elegir A", key=f"A_{index}",
                                     help="Funciona a veces"):
                choice_clicked = "Partial"

        # -------- Opción B ----------------------------------------------------
        with colB:
            st.markdown("### <span style='color:#FF5733;'>Opción&nbsp;B</span>",
                        unsafe_allow_html=True)
            st.markdown(
                f"**LOTERÍA:** **{p_guess*100:.0f}%** de probabilidad de que el "
                f"dispositivo funcione de forma fiable **todo el día**, y "
                f"**{(1-p_guess)*100:.0f}%** de que **NO funcione**."
            )
            if st.container().button("Elegir B", key=f"B_{index}",
                                     help="Lotería"):
                choice_clicked = "Lottery"

        # -------- Indiferente -------------------------------------------------
        with colC:
            st.markdown("### <span style='color:#AAAAAA;'>Indiferente</span>",
                        unsafe_allow_html=True)
            st.write("Aceptaría *cualquiera* de las dos opciones con estas probabilidades.")
            if st.container().button("Indiferente", key=f"C_{index}"):
                choice_clicked = "Indifferent"

        # -------- Lógica tras la selección -----------------------------------
        if choice_clicked is None:
            return  # nada pulsado

        if choice_clicked == "Partial":
            st.session_state[k_min] = p_guess
        elif choice_clicked == "Lottery":
            st.session_state[k_max] = p_guess
        elif choice_clicked == "Indifferent":
            st.session_state.responses_sg.setdefault(rid, {})[device_name] = p_guess * 100
            st.session_state.page_index_sg += 1
            st.rerun()
            return
    
        # update new midpoint & rerun same page
        st.session_state[k_guess] = (st.session_state[k_min] + st.session_state[k_max]) / 2
        st.rerun()

#---------------------------------------- Summary SG ------------------------------------------
    
    def sg_summary_page():
        st.title("Resumen de todos los dispositivos")
        rid = st.session_state.this_respondent_id
        this_resp_dict = st.session_state.responses_sg.get(rid, {})

        if not this_resp_dict:
            st.write("No se han registrado respuestas.")
        else:
            sorted_pairs = sorted(this_resp_dict.items(),
                                  key=lambda pair: pair[1],
                                  reverse=True)
            st.write("Utilidades seleccionadas (de mayor a menor):")
            for dev, util in sorted_pairs:
                st.write(f"• {dev}: {util:.3f}")

        if st.button("Ha terminado la encuesta. ¡Gracias!"):
            # Reiniciar para la siguiente persona encuestada
            st.session_state.page_index_sg = 0
            for dev in dev_load_map:
                for suffix in ("p_min", "p_max", "p_guess"):
                    st.session_state.pop(f"{dev}_{suffix}", None)
#            st.session_state.page_index = 6   # saltar al método PC
            finish_current_respondent()   
            st.rerun()
            return

#------------------------------------------ SG Menu ---------------------------------------

    if page_sg  == 0:
        sg_intro_page()                                      #We show the intro page
    elif page_sg == -1:
        sg_example_page()
    elif 1 <= page_sg <= total_devices:                      #We repeat to obtain probabilities for each device
        sg_interactive(page_sg)
    else:
        sg_summary_page()                                    #When the total number of devices is reached, we show the summary

################################ Pairwise Comparison ###########################################

def pairwise_method():                                     #We start the method
    page_pc = st.session_state.page_index_pc
    total_devices = len(dev_load_map)

# -------------------------------------- Helpers -------------------------------------
    
    def deduction(wins_pc, a, b, visited=None):
        #Return True if the graph already implies that a beats b, (transitively).
        #Deduction will help us return True if we can deduce from the wins dictionary (which
        #contains every device that "a" beats directly) that a > b (a and b are the devices we will
        #check) because of transitivity, taking the shortcut when possible (no asking if possible). 
             
        if visited is None:                 #If there is no visited set, we create it.
            visited = set()
        
        if a == b:                  #If a is equal to b, we do not consider that branch of the tree, it is not possible
            return False
        if b in wins_pc[a]:         #If b is in the branch of a, we define a wins b, and we add a to the visited set of b
            return True
            
        visited.add(a)
        #we take the devices inside the branch of a, and we check if it has been visited (we check in the set); if not, we apply deduction between x and b to skip the question
        for x in wins_pc[a]:
            if x not in visited:
                if deduction(wins_pc,x,b,visited):
                    return True
        return False

    def transitivity(wins_pc,a,b):
    # Ensure graph is transitively closed after adding winner → loser edge.
    # If a > b, propagate that knowledge in adjacency 'wins' (wins dict). If x is in wins[b], it should be added to wins [a]. For y in wins, if b in wins[y], we should add a to wins[y] and apply transitivity(y,a)
        if b != a and b not in wins_pc[a]:
            wins_pc[a].add(b)
            
        for x in wins_pc[b]:           #We apply transitivity, adding the devices b wins, to the wins dict of a.
            if x != a and x not in wins_pc[a]:
                wins_pc[a].add(x)
                transitivity(wins_pc, a, x)
        for y in list(wins_pc.keys()): #Here, we study if y could be > b... if a>b, we assume y>b (if y>a), because we know a>b
            if y == a:                 #Skip if y == a to avoid flipping the preference back onto b
                continue
            if a in wins_pc[y] and b not in wins_pc[y] and y != a:
                wins_pc[y].add(b)
                transitivity(wins_pc, y, b)

    def pick_next_pair(wins_pc, dev_load_map, checked_pairs_pc):   #Take next pair, not deducible by the transitivity function, and not yet asked.
        n = len(dev_load_map)
        for i in range(n):
            for j in range(i+1, n):
                A=dev_load_map[i]
                B=dev_load_map[j]
                if (A,B) in checked_pairs_pc or (B,A) in checked_pairs_pc:    #skip if it was already asked or if it can be deduced by transitivity
                    continue
                if deduction(wins_pc, A, B) ^ deduction(wins_pc, B, A):
                    #st.write('deduction was applied')
                    continue
                return (A,B)                                                  #if it is not deducible or yet asked, it is returned to be asked.
        return None

    def topological_sort(wins_pc):
        #Return devices from highest to lowest based on DFS topological sort.
        #Topological creates a visited dict where all the devices already visited are recorded so as not to become an infinite loop, because we 
        #append "u" after visiting its successors. Then, the DFS (depth-first search) is applied. This function takes a node (device) and visits 
        #all the possible successors (which are all the devices included in the wins dict for that specific device). If V has not been visited 
        #yet (the successor), we first explore all its successors. So we are running through the whole decision tree."""
        
        visited = {}
        order = []
    
        def dfs(u):
            visited[u] = True
        #For each device that 'u' beats, do DFS if not visited
            for v in wins_pc[u]:
                if not visited.get(v, False):
                    dfs(v)
        #Post-order: after exploring children, append u
            order.append(u)
    
    #We run DFS from every device that has not been visited yet
        for device in wins_pc:
            if not visited.get(device, False):
                dfs(device)
                
        #Because we append 'u' after all v in wins[u], "u" ends up to the right of v in the order list (right is worse than left). So 
        #"order" will be from "lowest rank" to "highest" if read left(high) to right (low). We want "highest first", so we reverse it:

        order.reverse()
        return order
        
# ------------------------------------- Intro Page -------------------------------------

    def pc_intro_page():
        st.title("Comparación por Pares – introducción rápida")

        st.markdown(
            """
            ¡Bienvenido/a al método de **_Comparación por Pares (PC)_**!  
            Aquí elegirá repetidamente cuál de **dos dispositivos** es más importante para usted.

            ### 🌩️ Contexto
            * El centro no puede alimentar todos los dispositivos a la vez.
            * Tu tarea es escoger el que debería recibir electricidad **ahora mismo**.

            ### 🔑 Puntos clave a recordar
            * **Si se elige el Dispositivo A ⇒ el Dispositivo B permanece apagado**
              (y viceversa).  
            * Existen *muchos* pares, pero la lógica inteligente omite
              comparaciones que ya podemos deducir.
            * Responde con **coherencia** – no hay límite de tiempo.

            ---
            """,
            unsafe_allow_html=True,
        )

        st.markdown("Cuando esté listo/a, haga clic en el botón ¡Empecemos!")

        if st.button("Comenzar comparación"):
            st.session_state["wins_pc"]          = {d: set() for d in dev_load_map}
            st.session_state["checked_pairs_pc"] = set()
            st.session_state.page_index_pc       = 1
            st.rerun()

#It takes the first pair not asked or deducible, and asks about it. The answer is recorded, and transitivity is applied. If there are no more pairs available, it shows the final ranking.

# ----------------------------------- Pairwise Comparison Page (one per pair) ---------------------------------

    def pairwise_page():
        st.title("Método de Comparación por Pares")
        st.subheader(f"Participante {st.session_state.this_respondent_id}")

        wins_pc       = st.session_state["wins_pc"]
        checked_pairs = st.session_state["checked_pairs_pc"]
        devices       = [d for d in dev_load_map]

        # Pares pendientes no deducibles ni preguntados
        def is_undecided(a, b):
            return not deduction(wins_pc, a, b) and not deduction(wins_pc, b, a)

        all_pairs = list(combinations(devices, 2))
        remaining = sum(
            1
            for (A, B) in all_pairs
            if (A, B) not in checked_pairs
            and (B, A) not in checked_pairs
            and is_undecided(A, B)
        )
        st.text(f"Preguntas restantes (máx.): {remaining}")

        pair = pick_next_pair(
            st.session_state["wins_pc"],
            dev_load_map,
            st.session_state["checked_pairs_pc"],
        )

        if pair is None:
            st.write(
                "No hay más pares. Todas las comparaciones están resueltas o deducidas. "
                "Esta es su clasificación:"
            )
            show_final_ranking(st.session_state["wins_pc"])

            # Guardar y avanzar
            rid = st.session_state.this_respondent_id
            st.session_state.responses_pc[rid] = topological_sort(
                st.session_state["wins_pc"]
            )

            if st.button("Finalizar este método"):
                for k in ("page_index_pc", "wins_pc", "checked_pairs_pc"):
                    st.session_state.pop(k, None)
#                finish_current_respondent()
                # al terminar PC pasamos a SG
                st.session_state.page_index_pc = 0
                st.session_state.page_index    = 5      # Standard Gamble
                st.rerun()
                return
        else:
            A, B = pair
            preference = st.radio(
                f"Si solo pudiera disponer de **uno** de estos dos dispositivos: "
                f"**{A}** o **{B}**, funcionando en su centro, ¿cuál elegiría?",
                [A, B],
            )

            if st.button("Enviar elección"):
                # Registrar par preguntado
                st.session_state["checked_pairs_pc"].add((A, B))

                if preference == A:
                    transitivity(st.session_state["wins_pc"], A, B)
                elif preference == B:
                    transitivity(st.session_state["wins_pc"], B, A)

                st.rerun()
#            st.write('After this pick: ',st.session_state["wins_pc"])

# ----------------------------------------- Summary Page -----------------------------------------

    def show_final_ranking(wins_pc):
        rid     = st.session_state.this_respondent_id
        ranking = topological_sort(wins_pc)  # de mayor a menor

        st.markdown(f"### Participante **{rid}**")
        st.markdown(
            "**Clasificación final**  \n"
            "_(1 = más importante → {n} = menos)_".format(n=len(ranking))
        )

        if not ranking:
            st.info("No se ha generado ninguna clasificación — revisa las respuestas.")
            return

        # Calcular utilidades lineales 100 ↘ 0
        util_map = normalise_answer("PC", ranking)

        col_rank, col_util = st.columns([4, 2])
    
        for pos, dev in enumerate(ranking, start=1):
            col_rank.markdown(f"{pos}. **{dev}**")
            col_util.markdown(f"{util_map[dev]:.1f}")

        st.caption(
            "Las utilidades se escalan linealmente de manera que #1 = 100 "
            "y el último = 0."
        )

# ----------------------------------------- PC Menu -----------------------------------------        

    if page_pc  == 0:
        pc_intro_page()
    elif 1 <= page_pc <= total_devices:
        pairwise_page()

################################################################################
# Final save / navigation helpers                                              #
################################################################################

# --------------------------------Página “gracias” --------------------------------

def thank_you_page() -> None:
    """Mensaje final para el encuestado."""
    st.markdown(
        """
        <h1 style='text-align:center; font-size:3rem; margin-top:3rem;'>
            ¡Gracias por participar!
        </h1>
        <h3 style='text-align:center; font-weight:normal; margin-top:2rem;'>
            Ya puede cerrar esta ventana.
        </h3>
        """,
        unsafe_allow_html=True,
    )
# --------------------------------------------------------------

def finish_current_respondent():
#    """
#    • Writes one JSON file per respondent to DATA_DIR  
#    • Updates the in-memory list `st.session_state.survey_data`  
##    • Checks whether the target sample size has been reached; if so,
#      flips the “finished” flag in survey_meta and jumps to the analytics page.
#   """

    rid = st.session_state.this_respondent_id.strip()
    
        # ---------- build record ------------------------------------------------
    record = {
        "id": rid,
        "Methods": {
            "SG": {
                "utility": normalise_answer(
                    "SG", st.session_state.responses_sg.get(rid, {})
                )
            },
            "PC": {
                "utility": normalise_answer(
                    "PC", st.session_state.responses_pc.get(rid, [])
                )
            }
        }
    }

# ---------- save to a respondent-specific file --------------------------
    out_path = DATA_DIR / RESP_PATTERN.format(rid=rid)
    out_path.write_text(json.dumps(record, indent=2))

    # ── update in-session data ─────────────────────────────────────────────
    st.session_state.completed_ids.add(rid)
    st.session_state.survey_data.append(record)

    # ── quota reached?  jump to optimisation-setup (page 98) ───────────────
    meta = st.session_state.survey_meta
    target_n = meta.get("target_n")              # may be None on first run

    if (
        target_n is not None
        and len(st.session_state.completed_ids) >= target_n
        and not meta.get("finished", False)
    ):
        meta["finished"] = True
        save_meta(meta)
#        st.session_state.page_index = 98         # optimisation setup
#        st.rerun()                  # stop here & redraw
                                   # ✂️  no code below runs
        
    # ── clean transient keys ───────────────────────────────────────────────
    for k in list(st.session_state.keys()):
        if k.endswith(("_sg", "_pc", "_es")) or k in {
            "page_index_sg", "page_index_pc", "wins_pc", "checked_pairs_pc",
        }:
            st.session_state.pop(k, None)

    # ── organiser options (only shown while quota not yet met) ─────────────
    st.info("Respondent saved.")

    st.session_state.this_respondent_id = None
    st.session_state.page_index = 120
    st.rerun()   

################################################################################
#  Summary page                                                                #
################################################################################

def final_summary():
    st.title("All respondents complete!")
    st.write("Survey data collected:")
    for respondent in st.session_state.survey_data:
        rid = respondent['id']
        st.markdown(f"### Respondent ID: {respondent['id']}")
        for method, data in respondent["Methods"].items():
            st.markdown(f"**Method:** {method}")
            for dev, util in data["utility"].items():
                st.write(f"• {dev}: {util:.1f}")
#            st.write("**Utilities:**")
#            st.json(data["utility"])
#            st.write("**Ranking:**")
#            st.write(data["rank"])
#        st.json(st.session_state.survey_data)

# -------------------- Helpers for folders and for optimization --------------------------------

OUTDIR = Path("outputs")
OUTDIR.mkdir(exist_ok=True)

def save_chart(chart: alt.Chart, stem: str):
    png = OUTDIR / f"{stem}.png"
    svg = OUTDIR / f"{stem}.svg"
    chart.save(png, scale=2, engine="vl-convert")   # tell Altair which backend
    chart.save(svg, engine="vl-convert")

def knapsack_dp(weights, values, capacity):
    """0-1 knapsack via dynamic programming – returns a 0/1 list."""
    weights = [int(round(w)) for w in weights]
    capacity = int(round(capacity))
    n = len(weights)
    dp = [[0]*(capacity+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for w in range(capacity+1):
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i-1][w],
                               values[i-1] + dp[i-1][w-weights[i-1]])
            else:
                dp[i][w] = dp[i-1][w]
    take = [0]*n
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            take[i-1] = 1
            w -= weights[i-1]
    return take

def run_optimisation(util_dict, power_map, P):
    #Return a dataframe with LP & DP selections and some totals.
    
    df = pd.DataFrame({
        "Device": list(util_dict),
        "Utility": [util_dict[d]  for d in util_dict],
        "Power":   [power_map[d] for d in util_dict]
    })
    df["Utility_per_Watt"] = df["Utility"] / df["Power"] 
    # ----------------------------- relaxed LP ----------------------------------------------
    c = -df["Utility"].to_numpy()
    res = linprog(c,
                  A_ub=[df["Power"].to_numpy()],
                  b_ub=[P],
                  bounds=[(0,1)]*len(df),
                  method="highs")
    df["LP_pick"] = np.round(res.x).astype(int)

    # ------------------------------ exact 0-1 DP ----------------------------------------------
    df["DP_pick"] = knapsack_dp(df["Power"].tolist(),
                                 df["Utility"].tolist(), P)

    # totals for convenience
    for col in ("LP_pick","DP_pick"):
        df[f"{col}_power"]   = df["Power"]   * df[col]
        df[f"{col}_utility"] = df["Utility"] * df[col]
    return df

# --------------------------- Analytics --------------------------------------

#Password helper 

def is_admin():
    """Devuelve True si el usuario ha introducido la contraseña correcta."""
    pwd_entered = st.session_state.get("admin_pwd", "")
    stored_pwd  = st.secrets.get("admin", {}).get("password", "")
    return pwd_entered and pwd_entered == stored_pwd
    
#-------------------------
def analytics_page():
    st.title("📊 Survey analytics")

    if not is_admin():
        st.subheader("🔐 Acceso a analytics")

        pwd_try = st.text_input(
            "Introduce la contraseña de analytics:",
            type="password",
            key="pwd_try"                # clave temporal
        )

        if st.button("Entrar"):
            # Guardamos el intento en session_state
            st.session_state.admin_pwd = pwd_try

            if is_admin():               # correcto → recargar página
                st.rerun()
            else:                        # incorrecto → mensaje de error
                st.error("❌ Contraseña incorrecta")

        st.stop()        # corta aquí hasta que la clave sea válida
        
    else:
        # 1.  Always load the latest JSONs from disk
        st.session_state.survey_data = load_all_responses()
        meta = st.session_state.survey_meta
    
        # 2.  Block access until the target sample size is done
        if not meta.get("finished"):
            st.warning(
                "Still waiting for respondents – optimisation & analytics will "
                "unlock automatically when the last questionnaire is complete."
            )
    
            if st.button("Keep on gathering data"):
                st.session_state.page_index = 2
                st.rerun()
            return 
    
        # 3.  ───────────────────── build long-form dataframe ────────────────────
        rows = []
        for rec in st.session_state.survey_data:
            rid = rec["id"]
            for method, block in rec["Methods"].items():
                for dev, util in block["utility"].items():
                    rows.append(
                        {
                            "Respondent": rid,
                            "Method": method,
                            "Device": dev,
                            "Utility": util,
                        }
                    )
    
        df = pd.DataFrame(rows)
        if df.empty:
            st.info("No data found on disk – please check your respondent files.")
            return
        #---------------------------- overall metrics (all methods together)----------------------------
        st.header("Overall (all methods combined)")
    
        #---------------------------- 1-rank counts --------------------------------
        top1_counts = (
            df
            .loc[df.groupby(["Respondent", "Method"])["Utility"].idxmax()]
            .groupby("Device")["Utility"]
            .size()
            .rename("Top-1 count")
            .reindex(dev_load_map, fill_value=0)
        )
    
    #    st.subheader("How often is each device ranked #1?")
    #    st.dataframe(top1_counts.to_frame())   # tabular view
        st.altair_chart(
            alt.Chart(top1_counts.reset_index(),
                      title="Frequency of being ranked #1").mark_bar().encode(
                x="Top-1 count:Q",
                y=alt.Y("Device:N", 
                        sort="-x",
                        axis=alt.Axis(title=None, labelLimit=0, labelPadding=6))
            ), use_container_width=True
        )
    
        #---------------------------- mean utilities ------------------------------------------
            # 1.  Series → sorted (highest-first)
        mean_util_ser = (
            df.groupby("Device")["Utility"]
              .mean()
              .sort_values(ascending=False)         
        )
        
        # 2.  Nice table (already sorted) ─────────────────────────────────────
        st.subheader("Average utility per device (0–100 %)")
        st.dataframe(mean_util_ser.round(2).to_frame(name="Average utility"))
        
        # 3.  Bar-chart with full labels, same order ──────────────────────────
        mean_util_df = (
            mean_util_ser.reset_index()              # columns → Device, Utility
                          .rename(columns={"Utility": "Average utility"})
        )
        
        st.altair_chart(
            alt.Chart(mean_util_df, title="Mean utility")
               .mark_bar()
               .encode(
                   x="Average utility:Q",
                   y=alt.Y(
                       "Device:N",
                       sort=mean_util_ser.index.tolist(),     # keep the same order
                       axis=alt.Axis(
                           title=None,        # ← remove the “Device” title
                           labelLimit=0,      # show full names
                           labelPadding=4     # tiny gap from the bars
                       )
                   )
               ),
            use_container_width=True,
        )
    
        # ----------------------------- per-method breakdown -------------------------------
        st.header("Method comparison")
        
        # -------------------------- 1. plain-text winners ------------------------------------
        overall_winner = top1_counts.idxmax()
        sg_winner      = top1_counts.loc[dev_load_map]  # reuse but filter below
        pc_winner      = top1_counts.loc[dev_load_map]
        
        sg_winner = (
            df[df["Method"] == "SG"]
              .groupby("Respondent")["Utility"].idxmax()
              .map(df.loc[:, "Device"])
              .value_counts()
              .idxmax()
        )
        
        pc_winner = (
            df[df["Method"] == "PC"]
              .groupby("Respondent")["Utility"].idxmax()
              .map(df.loc[:, "Device"])
              .value_counts()
              .idxmax()
        )
        
        st.markdown(
            f"* **Overall #1 device:** {overall_winner}\n"
            f"* **SG #1 device:** {sg_winner}\n"
            f"* **PC #1 device:** {pc_winner}"
        )
        
        # --------------------- 2. combined utility bar-chart ----------------------------
        # prepare long form dataframe with a colour label
        combo = (
            df.groupby(["Method", "Device"])["Utility"]
              .mean()
              .reset_index()
        )
        
        # two charts with identical scale concatenated left-right
        chart_sg = (
            alt.Chart(combo[combo.Method == "SG"])
                .mark_bar(color="#1f77b4")           # blue
                .encode(
                    x=alt.X("Utility:Q", scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y("Device:N", sort=dev_load_map)
                )
                .properties(title="SG mean")
        )
        
        chart_pc = (
            alt.Chart(combo[combo.Method == "PC"])
                .mark_bar(color="#d62728")           # red
                .encode(
                    x=alt.X("Utility:Q", scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y("Device:N", sort=dev_load_map)
                )
                .properties(title="PC mean")
        )
        
        st.altair_chart(alt.hconcat(chart_sg, chart_pc), use_container_width=True)
        
        # ---------------------- slope chart -------------------------------
        util_tbl = (
        df.groupby(["Method", "Device"])["Utility"]
          .mean()
          .unstack("Method")          # columns: SG, PC
          .reindex(dev_load_map)
          .reset_index()
        )
    
        bullet_base = alt.Chart(util_tbl).encode(
        y=alt.Y("Device:N", sort=dev_load_map, title=None),
        color="Device:N"
        )
    
        bullet_lines = bullet_base.mark_line().encode(
            x=alt.X("SG:Q", scale=alt.Scale(domain=[0, 100]),
                    axis=alt.Axis(title="Utility (%)")),
            x2="PC:Q"
        )
    
        sg_dots = bullet_base.mark_point(filled=True, size=70).encode(x="SG:Q")
        pc_dots = bullet_base.mark_point(filled=False, size=70, strokeWidth=2).encode(x="PC:Q")
        
        util_chart = (bullet_lines + sg_dots + pc_dots).properties(
            title="Mean utility – SG (●)  →  PC (○)",
            width=620
        )
    
        st.altair_chart(util_chart, use_container_width=True)
        save_chart(util_chart, "utilities_sg_pc")
    
        # ------------------- crossover ranking chart -----------------------------
        # build a “long” table: one row per device × side
        st.markdown("**PC → SG**".format(n=len(dev_load_map)))
        st.markdown("**Rank: 1 (top) → {n} (bottom)**".format(n=len(dev_load_map)))
        
        # 1. Compute ranks
        rank_sg = (
            df[df.Method=="SG"]
              .groupby("Device")["Utility"].mean()
              .rank(ascending=False, method="first")
              .astype(int)
        )
        rank_pc = (
            df[df.Method=="PC"]
              .groupby("Device")["Utility"].mean()
              .rank(ascending=False, method="first")
              .astype(int)
        )
        
        # 2. Build long-form DataFrame
        cross_df = pd.DataFrame(
            [{"Device": d, "Side":"PC", "x": 0, "rank": rank_pc[d]} for d in dev_load_map] +
            [{"Device": d, "Side":"SG", "x": 1, "rank": rank_sg[d]} for d in dev_load_map]
        )
        
        # 3. Base chart: hide both axes
        base = alt.Chart(cross_df).encode(
            x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[0,1])),
            y=alt.Y("rank:Q",
                    axis=None,
                    scale=alt.Scale(domain=[0.5, len(dev_load_map)+0.5], reverse=True))
        )
        
        # 4. Lines + points colored by device
        lines = base.mark_line(strokeWidth=1.5).encode(
            detail="Device:N",
            color=alt.Color("Device:N", legend=None)
        )
        points = base.mark_point(size=80, filled=True).encode(
            color=alt.Color("Device:N", legend=None)
        )
        
        # 5. Left labels (PC side) and right labels (SG side)
        labels_left = (
            base.transform_filter("datum.Side == 'PC'")
                .mark_text(align="right", baseline="middle", dx=-10)
                .encode(text="Device:N", color=alt.Color("Device:N", legend=None))
        )
        labels_right = (
            base.transform_filter("datum.Side == 'SG'")
                .mark_text(align="left", baseline="middle", dx=10)
                .encode(text="Device:N", color=alt.Color("Device:N", legend=None))
        )
        
        # 6. Compose & render
        rank_shift = (
            (lines + points + labels_left + labels_right)
              .properties(width=600, height=25 * len(dev_load_map))
              .configure_view(stroke=None)
        )
        st.altair_chart(rank_shift, use_container_width=True)
        save_chart(rank_shift, "rank_crossover")
    
        # --------------------- energy-budget optimisation -------------------------
        st.header("Optimised device bundle")
        
        if st.session_state.max_power is None:
            st.info("Maximum power not set – configure it on the setup page.")
            return
        
        # Build one utility number per device according to the survey-taker’s choice
        choice = st.session_state.utility_source   # "PC", "SG", "Average"
    
        if choice == "Average":
            util_tbl = df.groupby("Device")["Utility"].mean()
        else:
            util_tbl = (
                df[df.Method == choice]
                  .groupby("Device")["Utility"].mean()
            )
        
        # ----- Keep only devices that are present in the facility, then rescale -----
        avail_set = st.session_state.facility_devices
        util_opt  = filter_and_rescale_for_optim(util_tbl, avail_set, renorm=True)
        
        if util_opt.empty:
            st.warning("No available devices selected on the availability page!")
            return                # nothing to optimise
        
        # Note: util_opt is now 0-to-1; LP/DP don’t care about the scale.
        opt_df = run_optimisation(util_opt.to_dict(), power_map, st.session_state.max_power)
    
        opt_df["Utility01"]           = opt_df["Utility"] / 100        # 0-1 per device
        opt_df["Utility01_per_Watt"]  = opt_df["Utility01"] / opt_df["Power"]
    
                # helper to build text + table for one solver
        def bundle_summary(tag, flag_col, colour):
            avail_cnt = len(st.session_state.facility_devices)
            sel = opt_df[opt_df[flag_col] == 1][["Device", "Utility", "Power"]]
            sel_cnt = len(sel)
            sel.index = np.arange(1, sel_cnt + 1)
            tot_p = sel["Power"].sum()
            tot_u = sel["Utility"].sum()
            spare = st.session_state.max_power - tot_p
            headline = (
                f"From **{avail_cnt}** devices in the facility, "
                f"this critical-load set contains **{sel_cnt}** devices."
            )
            st.markdown(
                headline + "<br>" +
                f"**{tag} solution** &nbsp; "
                f"total power **{tot_p:.0f} W** / {st.session_state.max_power} W "
                f"({'{:+.0f}'.format(spare)} W spare)  &nbsp;|&nbsp; "
                f"total utility **{tot_u:.1f}**",
                unsafe_allow_html=True
            )
            st.table(sel.style.applymap(
                lambda _: f"background-color:{colour}; color:white")
            )
            st.markdown("---")
    
        # ------------------------ textual sumary ----------------------------
        pow_lp = int(opt_df["LP_pick_power"].sum())
        pow_dp = int(opt_df["DP_pick_power"].sum())
        util_lp = opt_df["LP_pick_utility"].sum()
        util_dp = opt_df["DP_pick_utility"].sum()
        
        st.markdown(
        f"*Capacity:* **{st.session_state.max_power} W** &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"**LP bundle:** {pow_lp} W → {util_lp:.1f} util &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"**DP bundle:** {pow_dp} W → {util_dp:.1f} util"
        )
        
        # ---------------------- Plot 1 power allocation ------------------
    
        sel_any = opt_df[(opt_df.LP_pick == 1) | (opt_df.DP_pick == 1)]    
        y = np.arange(len(sel_any))
        bar_height = 0.4
        
        st.subheader("Power allocation (blue = LP, orange = DP)")
        fig1, ax1 = plt.subplots(figsize=(7, 0.45*len(opt_df)))
        
        ax1.barh(y-bar_height/2, sel_any["LP_pick_power"], height=bar_height,
                 color="steelblue", label="LP")
        ax1.barh(y+bar_height/2, sel_any["DP_pick_power"], height=bar_height,
                 color="darkorange", alpha=.8, label="DP")
    
        lp_used = sel_any["LP_pick_power"].sum()
    
        ax1.axvline(st.session_state.max_power, ls="--", color="red",  label="Capacity")
        ax1.axvline(pow_dp, ls="--", color="darkorange",label="DP used")
        ax1.axvline(lp_used, ls="--", color="steelblue", label="LP used")
        
        ax1.set_yticks(y, sel_any["Device"])
        ax1.set_xlabel("Power (W)")
        ax1.legend(); 
        st.pyplot(fig1)
        
        # ------------------ Plot 2 utility per watt bars  ------------------
        st.subheader("Utility per Watt (selected devices coloured)")
        y2 = np.arange(len(opt_df))
        bar_height = 0.4
        
        fig2, ax2 = plt.subplots(figsize=(7, 0.45*len(opt_df)))
        ax2.barh(y2-bar_height/2, opt_df["Utility01_per_Watt"],
                 height=bar_height,
                 color=np.where(opt_df["LP_pick"], "steelblue", "#d0d0ff"))
        
        ax2.barh(y2+bar_height/2, opt_df["Utility01_per_Watt"],
                 height=bar_height,
                 color=np.where(opt_df["DP_pick"], "darkorange", "#ffd8b0"))
        
        ax2.set_yticks(y2, opt_df["Device"])
        ax2.set_xlabel("Utility (0–1) per W")                  
        st.pyplot(fig2)
    
        # ------------------ PLOT 3 – cumulative-utility curves ------------------
        st.subheader("Cumulative utility vs. power")
    
        # Build the “order” DataFrame if not already in scope
        order = opt_df.sort_values("Utility01_per_Watt", ascending=False).copy()
        order["cum_P"]   = order["Power"].cumsum()
        order["cum_U01"] = order["Utility01"].cumsum()
        
        # Identify the DP bundle point
        dp_P   = order.loc[order["DP_pick"] == 1, "Power"].cumsum().iloc[-1]
        dp_U01 = order.loc[order["DP_pick"] == 1, "Utility01"].cumsum().iloc[-1]
        
        # Create the figure
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        ax3.plot(order["cum_P"], order["cum_U01"],
                 marker="o", linestyle="-", color="steelblue",
                 label="Greedy order (rounding)")
        
        ax3.scatter(dp_P, dp_U01,
                    marker="^", s=100, color="darkorange",
                    label="DP")
        ax3.axvline(st.session_state.max_power,
                    ls="--", color="red", label="Capacity")
        
        # Collect Text objects
        texts = []
        for _, row in order.iterrows():
            txt = ax3.text(
                row.cum_P, row.cum_U01,
                row.Device,
                fontsize=8,
                ha="center", va="center"
            )
            texts.append(txt)
        
        # Let adjustText shove them apart
        adjust_text(
            texts,
            only_move={"text":"xy"},
            arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5),
            expand_text=(1.05, 1.2),
            expand_points=(1.05,1.2)
        )
        
        ax3.set_xlabel("Cumulative power (W)")
        ax3.set_ylabel("Cumulative utility (0–1)")
        ax3.legend(loc="lower right")
        st.pyplot(fig3)    
            
        # ------------------ PLOT 4 – sensitivity analysis ------------------
        st.subheader("Sensitivity: best utility vs. available power")
    
        # 1. Prepare data
        P_steps     = np.arange(200, st.session_state.max_power + 800, 200)
        best_lp, best_dp = [], []
        lbl_lp, lbl_dp   = [], []
        prev_lp_set      = set()
        prev_dp_set      = set()
        
        weights_int = opt_df["Power"].round().astype(int).tolist()
        values      = opt_df["Utility"].tolist()
        
        # 2. Compute at each capacity
        for P in P_steps:
            # — Greedy / LP approximation —
            cur_P = cur_U = 0
            added_lp = ""
            for _, row in order.iterrows():
                if cur_P + row["Power"] <= P:
                    if row["Device"] not in prev_lp_set:
                        added_lp = row["Device"]
                    cur_P += row["Power"]
                    cur_U += row["Utility"]
            best_lp.append(cur_U)
            lbl_lp.append(added_lp)
            if added_lp:
                prev_lp_set.add(added_lp)
        
            # — Exact 0-1 DP optimum —
            sel      = knapsack_dp(weights_int, values, P)
            mask     = np.array(sel, dtype=bool)
            cur_dp   = set(opt_df.loc[mask, "Device"])
            new_dp   = cur_dp - prev_dp_set
            added_dp = next(iter(new_dp)) if new_dp else ""
            best_dp.append(opt_df.loc[mask, "Utility"].sum())
            lbl_dp.append(added_dp)
            if added_dp:
                prev_dp_set.add(added_dp)
        
        # 3. Draw side-by-side subplots
        fig4, (ax_lp, ax_dp) = plt.subplots(1, 2, figsize=(12, 4),
                                           sharey=True, sharex=True)
        
        # — Greedy/LP plot —
        ax_lp.plot(P_steps, best_lp, "-o", color="steelblue", label="Greedy/LP")
        texts_lp = []
        for x, y, dev in zip(P_steps, best_lp, lbl_lp):
            if not dev:
                continue
            txt = ax_lp.text(x, y, dev,
                             fontsize=7, color="steelblue",
                             ha="left", va="bottom")
            texts_lp.append(txt)
        ax_lp.axvline(st.session_state.max_power, ls="--", color="red")
        ax_lp.set_title("Greedy/LP sensitivity")
        ax_lp.set_xlabel("Available power (W)")
        ax_lp.set_ylabel("Max utility achievable (0–1)")
        
        # declutter LP labels
        adjust_text(
            texts_lp,
            only_move={'text':'xy'},
            arrowprops=dict(arrowstyle='-', color='gray', alpha=0.3),
            expand_text=(1.02, 1.2),
            expand_points=(1.02, 1.2),
            ax=ax_lp
        )
        
        # — Exact DP plot —
        ax_dp.plot(P_steps, best_dp, "-^", color="darkorange", label="Exact DP")
        texts_dp = []
        for x, y, dev in zip(P_steps, best_dp, lbl_dp):
            if not dev:
                continue
            txt = ax_dp.text(x, y, dev,
                             fontsize=7, color="darkorange",
                             ha="left", va="top")
            texts_dp.append(txt)
        ax_dp.axvline(st.session_state.max_power, ls="--", color="red")
        ax_dp.set_title("Exact DP sensitivity")
        ax_dp.set_xlabel("Available power (W)")
        
        # declutter DP labels
        adjust_text(
            texts_dp,
            only_move={'text':'xy'},
            arrowprops=dict(arrowstyle='-', color='gray', alpha=0.3),
            expand_text=(1.02, 1.2),
            expand_points=(1.02, 1.2),
            ax=ax_dp
        )
        
        # 4. Shared legend & layout
        fig4.legend(loc="upper center", ncol=2, frameon=False)
        fig4.tight_layout(rect=[0, 0, 1, 0.94])
        st.pyplot(fig4)
    
        st.header("Device list chosen by each optimiser")
    
        bundle_summary("Relaxed-LP", "LP_pick", "#1f77b4")      # blue
        bundle_summary("0-1 DP",    "DP_pick", "#ff7f0e")       # orange
    
        if st.button("Change optimisation parameters"):
            st.session_state.page_index = 98
            st.stop()

################################################################################
#  Main Menu                                                                   #
################################################################################

# ───────────────────────────  Main dispatcher  ────────────────────────────
def main():
    """
    Flujo deseado (primera vez que se abre la app):

        0  → fijar tamaño muestral (target_n)
        1  → elegir dispositivos disponibles
        2  → pantalla de bienvenida del encuestado
        6  → PC
        5  → SG
      120  → “gracias”
    """
    page  = st.session_state.page_index        # valor actual

    # ── enrutado explícito ─────────────────────────────────────────────────
    if   page == 0:   survey_setup_page()
    elif page == 1:   device_availability_page()
    elif page == 2:   respondent_intro_page()
    elif page == 5:   standard_gamble_method()
    elif page == 6:   pairwise_method()
    elif page == 98:  optimisation_setup_page()
    elif page == 120: thank_you_page()
    else:             analytics_page()        # incluye el caso page == 99

if __name__ == "__main__":
    main()