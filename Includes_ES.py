
# Module overview
# This file implements a Streamlit survey app with three elicitation methods:
# SG (Standard Gamble), PC (Pairwise Comparison), and ES (Even Swap).
# The ES method is intended for local use only.

#Streamlit web‑app for eliciting healthcare utility preferences by means of three 
#methods: Standard‑Gamble (SG),  Pairwise Comparison (PC), and Even Swap (ES)

#A page index drives the navigation flow:

#    0 → survey setup (number of respondents)
#    1 → device availability checklist
#    2 → load‑type assignment for every device
#    3 → (reserved – method selection)
#    4 → respondent consent + ID
#    5 → Standard Gamble questionnaire
#    6 → Pairwise Comparison questionnaire
#   99 → final summary of all respondents

#The code is organised in self‑contained view functions, each responsible for 
#rendering one logical page and mutating the session state so that the main
#"main()" dispatcher can pick the next view.

import streamlit as st
import json
import pandas as pd
import numpy as np
import altair as alt
from collections import defaultdict
from pathlib import Path
from scipy.optimize import linprog
import matplotlib.pyplot as plt 
from itertools import combinations
from adjustText import adjust_text

################################################################################
#  Global constants                                                            #
################################################################################

dev_load_map = [ "Electric Stove", "Solar vaccine refrigerator", "Ultrasound","Oxygen Concentrator", "Electric Pots", "Gooseneck",
#               "Video Projector", "Electrocardiogram", "Nebulizer", "Dental Unit", "Refrigerator", "Electric Stretcher", "Secretion Aspirator",
#               "Sterilizer", "Infusion Pump", "Vital Signs Monitor", "Air Compressor", "Desktop", "Laptop", "Internet Connection", "Printer", 
#               "AC", "Light Bulbs"]
               ]
                

power_map = {
    "Electric Stove": 8000, "Solar vaccine refrigerator": 1104, "Ultrasound": 1158,
    "Oxygen Concentrator": 1180, "Electric Pots": 4000, "Gooseneck": 672,
#    "Video Projector": 32.5, "Electrocardiogram": 17.5, "Nebulizer": 120,
#    "Dental Unit": 7200, "Refrigerator": 576, "Electric Stretcher": 720,
#    "Secretion Aspirator": 220, "Sterilizer": 6000, "Infusion Pump": 200,
#    "Vital Signs Monitor": 1200, "Air Compressor": 5840, "Desktop": 720,
#    "Laptop": 180, "Internet Connection": 288, "Printer": 1200, "AC": 2700,
#    "Light Bulbs": 7920
}
# Session state bootstrap: defines navigation pointers, per-method pagers,
# response stores, and global configuration shared across views.

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

if "current_idx" not in st.session_state:            #0-based respondent index
    st.session_state.current_idx = 0                 

if "assignments" not in st.session_state:            # maps each device to a chosen load type (None = not chosen yet)
    st.session_state.assignments = {d: None for d in dev_load_map}

if "max_power" not in st.session_state:          # W – capacity of the system
    st.session_state.max_power = None
    
if "utility_source" not in st.session_state:     # "PC", "SG", or "Average"
    st.session_state.utility_source = None

if "page_index_es" not in st.session_state:  # per-respondent ES pager
    st.session_state.page_index_es = 0
    
if "chosen_devices_es" not in st.session_state:      # backpack content
    st.session_state.chosen_devices_es = []
    
if "remaining_kwh_es" not in st.session_state:       # energy left while packing
    st.session_state.remaining_kwh_es = None
    
if "responses_es" not in st.session_state:           # {rid: {dev: Pi*}}
    st.session_state.responses_es = {}

if "swap_counts" not in st.session_state: 
    st.session_state.swap_counts = {d:0 for d in dev_load_map}

################################################################################
#  Helper utilities                                                            #
################################################################################

def normalise_answer(method_code, answer, n=len(dev_load_map)):
    """Convert raw method outcomes into a comparable utility scale. SG is already 0–100%. PC is mapped linearly from rank to 0–100%. Returns a device→utility dict."""
    #Here, we will normalize the utilities between the different methods.

    #SG already delivers utilities in percent, so we simply copy them. 
    #PC gives the preferences in order; we map the first position to 100, the last to 0 via 
    #a linear scale.

    if method_code == "SG":
        return answer
    util={}
    for rank, dev in enumerate(answer, start=1):
        util[dev] = (n-rank)/(n-1)*100        # linear scale, first=100, last=0
    return dict(sorted(util.items(), key=lambda x: -x[1]))

    
def filter_and_rescale_for_optim(util_series, avail_set, renorm=True):
    """ Filter utilities to devices available in the facility and optionally rescale linearly to [0,1] for optimisation robustness."""
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
    """Streamlit view: configure global parameters (N respondents, power budget, utility source) and move to next page."""
    st.title("Survey Setup – For Survey Taker Only")
    
    num_resp = st.number_input("Number of respondents:", min_value=1, step=1)

    st.session_state.max_power = st.number_input("Power available for critical loads(W):",
                                                 min_value=100, step=100,
                                                 value=st.session_state.max_power or 10000
                                                )
    st.session_state.utility_source = st.radio("Which utilities do you want to use for optimization?",
                                               options=["SG", "PC","ES", "Average"],
                                               horizontal = True,
                                               index=(["PC","SG","ES","Average"].index(st.session_state.utility_source)
                                                      if st.session_state.utility_source else 3)
                                              )
    
    if st.button("Start"):
        st.session_state.num_respondents = num_resp
        #We create the list of IDs to check later for uniqueness
        st.session_state.page_index = 1 
        st.session_state.ids = [None]*num_resp

def device_availability_page():
    """Streamlit view: checklist to mark which devices exist in the current facility."""
    st.title("Device Availability - For Survey Taker Only")
    
    st.write("Please check which devices are available in the facility:")
    
    for dev in dev_load_map:
        chk = st.checkbox(dev, key=f"chk_{dev}")
        if chk:
            st.session_state.facility_devices.add(dev)
        else:
            st.session_state.facility_devices.discard(dev)
    
    if st.button("Confirm Devices"):
        # move on to the classification of the devices into the Load types
        st.session_state.page_index = 2 

################################################################################
#  Respondent‑level pages                                                      #
################################################################################

def respondent_intro_page():
    """Streamlit view: consent text and unique respondent ID input; advances to elicitation method selection."""
    st.title(f"Respondent Setup")

    st.write("Before introducing your ID and proceeding to the method, it is important that you read and understand the following information regarding the management of your data.")
    st.write("Place and time:                                             - To be introduced")
    st.write("Contact person if you have any questions:                   - Miguel Lacomba Albert, ETH Zürich Student, mlacomba@student.ethz.ch")
    st.write("Data Protection Officer ETH Zurich:		                  - Tomislav Mitar (tomislav.mitar@sl.ethz.ch)")
    st.write("----------------------------------------------------------------------------")
    st.write("We would like to ask you if you are willing to participate in our research project. Your participation is voluntary. Please read the text below carefully and ask the conducting person about anything you do not understand or would like to know.")
    st.write("**What is investigated and how?**")
    st.write("     This study investigates your preferences regarding powered medical devices and appliances through two structured in-person surveys. You will be presented with a hypothetical scenario involving limited electricity access and will respond to a series of questions using two standard preference elicitation techniques: Pairwise Comparison and Standard Gamble. Your responses will be used to derive utility values—representing the relative importance of each device—which will then serve as mathematical parameters in an optimization model. The goal is to identify the most critical electrical loads in healthcare settings based on these preferences.")
    st.write("**Who can participate?**")
    st.write("     To be eligible for participation, you must currently be employed in a healthcare facility and have familiarity with its day-to-day clinical operations. Participants must also be able to complete a 30-minute survey session on a computer.")
    st.write("**What am I supposed to do as a participant?**")
    st.write("     You will be asked to evaluate medical devices under different electricity availability scenarios and indicate your preferences. This involves responding to survey questions designed to capture your judgment about the criticality of each device.")
    st.write("**What are my rights during participation?**")
    st.write("     Participation in this study is entirely voluntary. You have the right to withdraw at any time without providing a reason and without any negative consequences.")
    st.write("**What risks and benefits can I expect?**")
    st.write("     There are no anticipated physical or psychological risks other than perhaps mild discomfort due to computer screen exposure and computer use. The study involves only answering two surveys, guided by the survey taker, and poses minimal inconvenience. Your participation contributes to research that may enhance the resilience of healthcare infrastructure in low-resource settings.")
    st.write("**Will I be compensated for my participation?**")
    st.write("     No financial or material compensation is provided for participation in this study.")
    st.write("**What data is collected from me and how is it used?**")
    st.write("     No personal identifying information will be collected. The study only records your survey responses, which are used to calculate utility scores for each device. These utilities help inform the optimization of critical load sets in healthcare facilities.")
    st.write("----------------------------------------------------------------------------")

    st.write("If you understand and agree with the mentioned above, please introduce the ID you will want to use and sign the consent form you were given in paper.")
    
    resp_id = st.text_input("Enter Respondent ID (unique):", key="resp_id_input")
    
    if st.button("Proceed to method"):
        if not resp_id.strip():
            st.warning("Please enter a valid ID.")
        elif resp_id in st.session_state.ids:
            st.warning("This ID is already in use, please choose a different one.")
            return
        #We store the ID
        st.session_state.this_respondent_id = resp_id
        st.session_state.ids[st.session_state.current_respondent_num-1] = st.session_state.this_respondent_id
        st.session_state.page_index = 5 #run selected methods

###################################### Choose the method for survey i ##############################################

def run_selected_method():
    """Streamlit view: radiobutton to choose SG / PC / ES for the current respondent and route to the method engine."""
    st.title("Choose elicitation method")

    METHODS = ["SG", "PC", "ES"]
    default = (METHODS.index(st.session_state.selected_method)
               if st.session_state.selected_method in METHODS else 0)

    choice = st.radio(
        f"Please pick the method for respondent "
        f"{st.session_state.this_respondent_id}:",
        options=METHODS,
        horizontal=True,
        index=default          # always a valid int
    )

    if st.button("Continue"):
        # remember the pick …
        st.session_state.selected_method = choice

        # … reset the per-method local pagers so each starts fresh
        st.session_state.page_index_sg = 0
        st.session_state.page_index_pc = 0
        st.session_state.page_index_es = 0

        # jump to the new screen that actually runs the chosen module
        st.session_state.page_index = 6
        

def respondent_method_page():                    #  page_index == 6
    """Launch the SG / PC / ES engine that the picker stored."""
    code = st.session_state.selected_method
    if code == "SG":
        standard_gamble_method()
    elif code == "PC":
        pairwise_method()
    elif code == "ES":
        even_swap_method()
    else:
        st.error("No method selected – please go back.")

###################################### Standard Gamble ##############################################

def standard_gamble_method():
    """Implements the SG method as a finite-state mini-workflow (intro → per-device interactive pages → summary). Stores per-respondent utilities."""
    page_sg = st.session_state.page_index_sg
    total_devices = len(dev_load_map)

#------------------------------------------- Intro Page -------------------------------------------
    
    def sg_intro_page():
        """ Intro screen for the Standard Gamble method."""
        st.title("Welcome to the Standard Gamble Preference Elicitation Survey!")
        st.write("SG is a method for preference elic. that helps evaluate risk. In this case, you will be deciding which devices should or shouldn't be powered. The situation is the following:")
        st.write('Sometimes, the power demanded by our healthcare facility is greater than that our facility can generate. In this case, you are now in charge of deciding or gambling (when you are unsure) which devices should or shouldn’t be powered. We would like to know whether you would prefer a guaranteed but partial amount of power for a specific device or a “gamble” that might yield full or no power.')
        st.write('You will be presented with devices which can and cannot be present in your facility. In each situation, you will have to choose between:')
        st.write('Partial: Meaning the partially covered scenario for this device is enough.')
        st.write('Lottery: You are willing to gamble in order to obtain maximum utility for that device.')
        st.write('Indifferent: the point where you do not consider moving the probabilities. You are okay with the shown utilities for that device.')
        st.write("**When you're ready**, click below to begin.")
        if st.button("Move on"):
            st.session_state.page_index_sg = 1                #Move to the first device

# ----------------------------------- Device Page (one per page) ---------------------------------
    
    def sg_interactive(index):
        """Single-device SG question page with ternary choice (Partial/Lottery/Indifferent) and bisection over probability."""
        st.title("Standard Gamble Preference Elicitation Framework")
        st.subheader(f"For Respondent {st.session_state.this_respondent_id}")
        device_name = dev_load_map[index-1]
        rid = st.session_state.this_respondent_id

        if rid not in st.session_state.responses_sg:          #Make sure the inner dictionary exists
            st.session_state.responses_sg[rid] = {}

        total_devices = len(dev_load_map)
        st.write(f"**Device {index} of {total_devices}**")    #Display the progress, current device over total number of devices

        if device_name in st.session_state.facility_devices:  #Distinguish if device is in facility or not (to watch the behaviour of the resp)
            st.write(f"**Device**: {device_name} (Available in your facility)")
            st.write("Scenario: ")
            st.write("This device will be powered but is UNRELIABLE during operation, which means it may suddenly shut down because it occasionally requires more power than is assigned to it, making it impossible for the facility to sustain.")
        else:
            st.write(f'**Device**: {device_name} (NOT Available in your facility)')
            st.write("**Hypothetical Scenario**")
            st.write("This device will be powered but is UNRELIABLE during operation, which means it may suddenly shut down because it occasionally requires more power than is assigned to it, making it impossible for the facility to sustain.")

        #We reset probabilities for each new device, so it always starts at the same point: 50% to partial and 50% to lottery. 
        #Initially: p_max at 1, p_min at 0 and p_guess at 0.5
        if f"{device_name}_p_min" not in st.session_state:
            st.session_state[f"{device_name}_p_min"] = 0.0
        if f"{device_name}_p_max" not in st.session_state:
            st.session_state[f"{device_name}_p_max"] = 1.0
        if f"{device_name}_p_guess" not in st.session_state:
            st.session_state[f"{device_name}_p_guess"] = 0.5

        p_min = st.session_state[f"{device_name}_p_min"]
        p_max = st.session_state[f"{device_name}_p_max"]
        p_guess = st.session_state[f"{device_name}_p_guess"]

        st.write(f"Lottery: **{p_guess*100:.2f}%** chance of FULL coverage, **{(1 - p_guess)*100:.2f}%** chance of NO coverage at all.")  
        #Show current p_guess, respondent chooses between three options
        choice = st.radio(
            "Which do you prefer?",
            ["Partial", "Lottery", "Indifferent"],
            key=f"choice_{index}"
        )
        
        button_label = "Submit Choice"                                       #The button changes when picks Indifferent
        if choice == "Indifferent":
            button_label = "Next Device"

        if st.button(button_label, key=f"button_{index}"):                    #When the respondent pushes the button: 
            if choice == "Partial":                                          #If Partial, we update the p_min to p_guess
                st.session_state[f"{device_name}_p_min"] = p_guess
            elif choice == "Lottery":                                        #If Lottery, we update the p_max to p_guess
                st.session_state[f"{device_name}_p_max"] = p_guess
            elif choice == "Indifferent":                                    #If Indif., p_guess is stored as current prob for device and resp.(rid)
                st.session_state.responses_sg[rid][device_name] = p_guess*100
                st.session_state.page_index_sg += 1                          #And we move to the next device
                return                                                       #We'll reset p_min, p_max, p_guess for next devices

            
            #Then, we recompute the new guess (depending on the option chosen)
            #If Partial, P_guess will increase because p_max stays the same and p_min increases, hence the average will increase.
            #If Lottery, P_guess will decrease because p_min stays the same and p_max decreases, hence the average will decrease.
            #If Indifferent, we will not get to this point.
            
            pm = st.session_state[f"{device_name}_p_min"]
            pM = st.session_state[f"{device_name}_p_max"]
            new_p = (pm + pM) / 2
            st.session_state[f"{device_name}_p_guess"] = new_p

#---------------------------------------- Summary SG ------------------------------------------
    
    def sg_summary_page():
        """Show SG utilities collected for all devices for this respondent."""
        st.title("Summary of All Devices")
        rid = st.session_state.this_respondent_id                                               # current respondent id
        this_resp_dict = st.session_state.responses_sg.get(rid, {})

        if not this_resp_dict:                 #We check if he/she has yet answered; if not, we show: no responses recorded
            st.write("No responses recorded.")

        else:                                  #If he/she has already answered, we sort his/her answers by utility (from highest to lowest) 
            sorted_pairs = sorted(
                this_resp_dict.items(),
                key=lambda pair: pair[1],      # pair = (device, utility)
                reverse=True
            )
            st.write("Here are the utilities you picked for each device (descending order):")    #We show sorted utilities
            for dev, util in sorted_pairs:
                st.write(f"• {dev}: {util:.3f}")

        if st.button("You have finished the survey. Thank you!"):
            st.session_state.page_index_sg = 0               #We reset the number of the page index
            for dev in dev_load_map:                         #We reset probabilities for the following respondents and devices
                for suffix in ("p_min", "p_max", "p_guess"):
                    key = f"{dev}_{suffix}"
                    st.session_state.pop(key, None)          #Ignore if missing
            finish_current_respondent() 

#------------------------------------------ SG Menu ---------------------------------------

    if page_sg  == 0:
        sg_intro_page()                                      #We show the intro page
    elif 1 <= page_sg <= total_devices:                      #We repeat to obtain probabilities for each device
        sg_interactive(page_sg)
    else:
        sg_summary_page()                                    #When the total number of devices is reached, we show the summary

################################ Pairwise Comparison ###########################################
#  ─────────────────────────────────────────────────────────────────────────
# Pairwise method: comparison graph + transitive closure to minimise questions.

def pairwise_method():                                     #We start the method
    """ Implements the PC method using a transitive tournament graph; deduces implied preferences and yields a topological ranking."""
    page_pc = st.session_state.page_index_pc
    total_devices = len(dev_load_map)

# -------------------------------------- Helpers -------------------------------------
    
    def deduction(wins_pc, a, b, visited=None):
        """Recursive reachability: does the current wins graph already imply a > b via transitivity?"""
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
        """Maintain transitive closure of the wins graph after adding an edge winner→loser."""
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
        """Return next (A,B) not yet asked and not deducible by transitivity, or None if all resolved."""
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
        """DFS-based topological ordering of devices from best to worst given the wins graph."""
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

    def pc_intro_page():                                   #Small intro for the survey respondent
        """ Intro screen for the Pairwise Comparison method."""
        st.title("Pairwise Comparison Preference Elicitation Method")
        st.write("Welcome to the Pairwise Comparison method!")
        st.write("In this survey, you just have to choose **(wisely)** which devices are more important for you between the options shown on the screen.")
        st.write("The situation your facility is facing is the following:")
        st.write("Imagine there is not enough generation in the healthcare facility to power all the available devices.")
        st.write("In that case, you will need to decide at each step of the survey regarding two devices. You should assign power based on how important you think each device is for the daily operation or, specifically, for that day’s functioning of the facility.")
        st.write("**Note:** choosing device A would mean not powering the device included in option B")
        st.write("When you feel familiar with the method and prepared, hit the button below!")   
        if st.button("Start Pairwise Survey"):
            st.session_state["wins_pc"] = {d: set() for d in dev_load_map}  # full list
            st.session_state["checked_pairs_pc"] = set()
            st.session_state.page_index_pc = 1  #We increase the number of the page index to move on in the survey

#It takes the first pair not asked or deducible, and asks about it. The answer is recorded, and transitivity is applied. If there are no more pairs available, it shows the final ranking.

# ----------------------------------- Pairwise Comparison Page (one per pair) ---------------------------------

    def pairwise_page():
        """Ask one undecided pair; record the choice and propagate transitive wins."""
        st.title('Pairwise Comparison Preference Elicitation Method')
        st.subheader(f"For Respondent {st.session_state.this_respondent_id}")
    
        wins_pc       = st.session_state["wins_pc"]
        checked_pairs = st.session_state["checked_pairs_pc"]
        devices       = [d for d in dev_load_map]
    
        # ---------- count remaining non-deducible, non-asked pairs -----------------
        def is_undecided(a, b):
            """See inline comments. Docstring placeholder."""
            return not deduction(wins_pc, a, b) and not deduction(wins_pc, b, a)
    
        all_pairs = list(combinations(devices, 2))
        remaining = sum(
            1 for (A, B) in all_pairs
            if (A, B) not in checked_pairs
               and (B, A) not in checked_pairs
               and is_undecided(A, B)
        )
        st.text(f"Questions remaining (max): {remaining}")
        
        pair = pick_next_pair(st.session_state['wins_pc'], dev_load_map, st.session_state['checked_pairs_pc'])
        if pair is None: 
            st.write("No next pair. All comparisons resolved or implied. Here’s your ranking:")
            show_final_ranking(st.session_state['wins_pc'])
            # Store and advance
            rid = st.session_state.this_respondent_id
            st.session_state.responses_pc[rid] = topological_sort(st.session_state['wins_pc'])
            if st.button("Finish this method"):
                for k in ("page_index_pc","wins_pc","checked_pairs_pc"):
                    st.session_state.pop(k, None)
                finish_current_respondent()
                return
        else:
            A,B = pair
            preference = st.radio(f'Which do you prefer, **{A}** or **{B}**?', [A, B])
#            st.session_state["wins_pc"]
            if st.button('Submit choice'): 
                #Checked pairs store the pair so we will not ask about it again
                st.session_state['checked_pairs_pc'].add((A,B))
                if preference == A:
                    #We apply transitity to write in the wins dict implied wins
                    transitivity(st.session_state['wins_pc'], A, B)
#                    st.write("Adjacency so far:", st.session_state["wins_pc"])
                elif preference == B:
                    transitivity(st.session_state['wins_pc'], B, A)
#                    st.write("Adjacency so far:", st.session_state["wins_pc"])
                else: 
                    pass
#            st.write('After this pick: ',st.session_state["wins_pc"])

# ----------------------------------------- Summary Page -----------------------------------------

    def show_final_ranking(wins_pc):
        """Display the final PC ranking for the respondent."""
        st.subheader(f"For Respondent {st.session_state.this_respondent_id}")
        st.write('**Final Ranking** (in descendent order, from most important to least important for you):')
        ranking = topological_sort(wins_pc)
        st.write(ranking)

# ----------------------------------------- PC Menu -----------------------------------------        

    if page_pc  == 0:
        pc_intro_page()
    elif 1 <= page_pc <= total_devices:
        pairwise_page()

################################################################################
# Even Swap Method                                                             #
################################################################################
# Even Swap method: local-only interactive trade-off engine.

def even_swap_method():
    """Implements the ES method: backpack selection under a power budget → iterative even-swap trade-offs → scaled utilities."""
    # ────────────── convenience aliases ─────────────────────────────────────
    page_es = st.session_state.page_index_es
    rid     = st.session_state.this_respondent_id   # shorthand

    st.session_state.setdefault("swap_counts",     {d: 0 for d in dev_load_map})
    st.session_state.setdefault("responses_es",    {})
    st.session_state.setdefault("pi_prev_choice_es", {})
    st.session_state.setdefault("pi_current_es",     {})
    st.session_state.setdefault("unused_pool_es",    set())
    
    # ─────────────────────────────────────────────────────── ES-0 – intro ───
    if page_es == 0:
        st.title("Welcome to the Even Swap Preference Elicitation Survey!")
        st.write("Even Swap is a method that, apart from allowing us to see which devices are more important for you, will also allow us to see how much important those devices are to you.")
        st.write("The situation is the following:")
        st.write("You will be presented to a total available amount of power, and with devices which can or cannot be present in your facility. Then you will be asked to, knowing the total energy budget you have, to choose which devices you want to include or have available in your facility, **WITHOUT EXCEEDING THE BUDGET**. Once you are comfortable with your decision, you will move on to the second part of the method.")
        st.write("In this second part, you will presented with the devices chosen initially **one by one**, and the following situation:")
        st.write("Would you be willing to:")
        st.write("Option $P_i$: You stop powering the device (so you free power), and you can change it for a combination (or a single device) of the non-initially chosen devices. Also, the future $P_i$ will be with a smaller value, to see if you still want to make a trade-off with your initially chosen device.")
        st.write("Option $Device_i$: You decide to keep powering the initially chosen device. $P_i$ will be increased in the next question to see how you behave.")
        st.write("When you feel prepared to strat the method, hit the Start Packing Button!")

        if st.session_state.max_power is None:
            st.error("⚠︎  Please set a power budget on the setup page first.")
            return

        st.info(f"Budget available: **{st.session_state.max_power:.0f} W**")
        if st.button("Start packing"):
            st.session_state.remaining_w_es = float(st.session_state.max_power)
            st.session_state.page_index_es  = 1
            # reset per-run state
            st.session_state.chosen_devices_es   = []
            st.session_state.pi_prev_choice_es   = {}
            st.session_state.pi_current_es       = {}
            st.session_state.unused_pool_es      = set(dev_load_map)

    # ───────────────────────────────────────────────── ES-1  Backpack ────────
    elif page_es == 1:
        st.subheader("Select devices to power (the backpack)")

        # show “Device ( W )” in the dropdown
        fmt = lambda d: f"{d}  ({power_map[d]:.0f} W)"

        pack = st.multiselect(
            "Backpack devices",
            options=dev_load_map,
            default=st.session_state.chosen_devices_es,
            format_func=fmt,
        )
        st.session_state.chosen_devices_es = pack

        used = sum(power_map[d] for d in pack)
        rem  = st.session_state.max_power - used
        st.info(f"Remaining capacity: **{rem:.0f} W**")

        if rem < 0:
            st.error("Over capacity – please remove something.")
        elif pack and st.button("Lock backpack"):
            # initial Pi is each device’s own power draw
            st.session_state.pi_current_es = {d: power_map[d] for d in pack}
            st.session_state.unused_pool_es = set(dev_load_map) - set(pack)
            st.session_state.current_pi_idx_es = 0
            st.session_state.page_index_es = 2

    # ───────────────────────────────────────────────── ES-2  Trade loop ──────
    elif page_es == 2:
        devices = st.session_state.chosen_devices_es
        idx     = st.session_state.current_pi_idx_es

        # finished all backpack devices?
        if idx >= len(devices):
            st.session_state.page_index_es = 3
            st.rerun()

        d   = devices[idx]                       # current device
        Pi  = st.session_state.pi_current_es[d]  # its current offer

        st.subheader(f"{idx+1}/{len(devices)} – {d}")
        st.write(f"**Current Pi = {Pi:.0f} W**")

        # ── rebuild basket widget state -------------------------------------
        basket_key  = f"basket_{d}_{Pi}"
        basket_prev = st.session_state.get(basket_key, [])

        used_P  = sum(power_map[x] for x in basket_prev)
        remain_P = Pi - used_P

        # candidate pool that still fits
        fresh = [u for u in st.session_state.unused_pool_es
                 if power_map[u] <= remain_P and u not in basket_prev]
        candidates = basket_prev + fresh                    # keep defaults!

        basket = st.multiselect(
            f"Pick the device(s) you would be willing to power instead of {devices[idx]} with the current Pi",
            options=candidates,
            default=basket_prev,
            format_func=lambda x: f"{x} ({power_map[x]:.0f} W)",
            key=basket_key,
        )

        used_P   = sum(power_map[x] for x in basket)
        remain_P = Pi - used_P
        colour   = "success" if remain_P == 0 else "info"
        getattr(st, colour)(
            f"Basket power: **{used_P:.0f} W** · capacity left: **{remain_P:.0f} W**"
        )

        colK, colT = st.columns(2)
        keep_btn  = colK.button("Keep original device", key=f"K_{d}_{Pi}")
        trade_btn = colT.button("Trade for the basket", key=f"T_{d}_{Pi}",
                                disabled = remain_P < 0)

        if not (keep_btn or trade_btn):
            return                                 # wait for a click

        choice = "K" if keep_btn else "T"
        prev   = st.session_state.pi_prev_choice_es.get(d)

        # ── first flip  →  Pi★ found, store & move on -----------------------
        if prev and prev != choice:
            st.session_state.responses_es.setdefault(rid, {})[d] = Pi
            st.session_state.current_pi_idx_es += 1
            st.session_state.pi_prev_choice_es.pop(d, None)
            st.rerun()

        # ── update Pi -------------------------------------------------------
        if choice == "T":                          # TRADE
            if not basket:
                st.warning("Select at least one device to trade.")
                return
            delta = np.mean([power_map[x] for x in basket])
            Pi_new = max(0, Pi - delta)
            # update win counts for outside utilities
            for x in basket:
                st.session_state.swap_counts[x] += 1
        else:                                      # KEEP
            pool  = st.session_state.unused_pool_es
            delta = np.mean([power_map[x] for x in pool]) if pool else 0
            Pi_new = Pi + delta

        st.session_state.pi_current_es[d] = Pi_new
        st.session_state.pi_prev_choice_es[d] = choice
        st.rerun()

    # ───────────────────────────────────────────────── ES-3  Summary ────────
    elif page_es == 3:
        st.title("Summary of your Pi★ trade-offs")

        pis = st.session_state.responses_es.get(rid, {})
        if not pis:
            st.error("No Pi★ values recorded – something went wrong.")
            return

        # 1) scale backpack devices  →  50–100
        p_min, p_max = min(pis.values()), max(pis.values())
        u_backpack = {}
        for dev, p in pis.items():
            if p_max == p_min:                    # all equal edge-case
                u_backpack[dev] = 100.0
            else:
                u_backpack[dev] = 50 + 50*(p - p_min)/(p_max - p_min)

        # 2) outside devices from swap frequency  →  0.1–49.9
        swaps   = st.session_state.swap_counts
        max_ct  = max(swaps.values()) if swaps else 0
        u_out = {}
        for dev, ct in swaps.items():
            if dev in u_backpack:                 # skip backpack devices
                continue
            u_out[dev] = 0.1 if ct == 0 else 49.9*ct/max_ct

        utilities = {**u_backpack, **u_out}

        for dev, u in sorted(utilities.items(),
                             key=lambda kv: kv[1], reverse=True):
            st.write(f"**{dev}** → utility **{u:.1f}**")

        if st.button("Finish this method"):
            st.session_state.utilities_current = utilities
            st.session_state.selected_method   = "ES"
            finish_current_respondent()
            
#    if page_es  == 0:
#        es_intro_page()
#    elif page_es == 1: 
#        es_backpack()
#    elif page_es == 2:
#        es_interactive()
#    elif page_es == 3:
#        es_summary_page()

################################################################################
# Final save / navigation helpers                                              #
################################################################################

def finish_current_respondent():
    """ Persist the current respondent’s data (per method) to JSON/TXT, reset per-method state, and advance to next respondent or summary."""
    #Persist answers of the current respondent and move to next one.
    idx   = st.session_state.current_idx
    rid   = st.session_state.this_respondent_id
    mcode = st.session_state.selected_method          #"SG" / "PC" / "ES"

    while len(st.session_state.survey_data) <= idx:   #We ensure the list is long enough
        st.session_state.survey_data.append({})

    rec = st.session_state.survey_data[idx]         # shorthand

    rec["id"]      = rid
    rec["method"]  = mcode
    rec["devices"] = list(st.session_state.facility_devices)

    if mcode == "SG":
        # Standard-Gamble already stores utilities as percentages
        raw        = st.session_state.responses_sg.get(rid, {})      # {dev: util}
        rank_list  = sorted(raw, key=raw.get, reverse=True)
        utilities  = raw                                            # keep as-is

    elif mcode == "PC":
        # Pairwise gives only an ordering (list)
        raw        = st.session_state.responses_pc.get(rid, [])      # [dev, …]
        rank_list  = raw
        utilities  = normalise_answer("PC", rank_list)               # 0-100 linear

    elif mcode == "ES":
        # ▸ ES utilities were computed in ES-3 and saved here
        utilities = st.session_state.get("utilities_current", {})
        raw       = st.session_state.responses_es.get(rid, {})   # Pi★ values

        # fallback – should not normally trigger
        if not utilities:
            rank_list = sorted(raw, key=raw.get, reverse=True)
            utilities = normalise_answer("ES", rank_list)

        # ---make sure rank_list exists in the normal path -------------
        if 'rank_list' not in locals():                          
            rank_list = sorted(utilities, key=utilities.get, reverse=True)

    # ------------------------------------------------------------------------
    # make sure *every* device appears – missing ones get utility 0.0
    for d in dev_load_map:
        utilities.setdefault(d, 0.1)

    # store everything we might need later (optimiser, audit, etc.)
    rec["raw"]       = raw
    rec["rank"]      = rank_list
    rec["utility"]   = utilities
    rec["device_and_utility"] = [
        {"device": d, "utility": utilities.get(d, 0.0)}
        for d in dev_load_map
    ]

    # ensure nested dict exists and store this method’s data
    meth_block = {"utility": utilities,
                  "rank":    rank_list,
                  "raw":     raw}
    rec.setdefault("Methods", {})[mcode] = meth_block
    
# --------------------- Files per respondent ------------------------
        
    filename = f"respondent_{rid}.json"
    with open(filename, "w") as f:
        json.dump(rec, f, indent=2)
    file_txt = f"resp_txt_{rid}.txt"
    txt_lines = [f"Respondent ID: {rid}\n"]
    for method, data in rec["Methods"].items():
        txt_lines.append(f"method: {method}")
        for dev, util in data["utility"].items():
            txt_lines.append(f"• {dev}: {round(util, 1)}")
        txt_lines.append("")  # Add a blank line between methods
        with open(file_txt, "w") as t:
            t.write("\n".join(txt_lines))
            
    st.write('Results saved to results_id.json')
    st.write('Results saved to results.txt')

# ----------------------- Clenaing variables -----------------------

    for key in list(st.session_state.keys()):
        if key.endswith("_sg") or key.endswith("_pc") or key.endswith("_es"):
            st.session_state.pop(key, None)

    st.session_state.current_idx += 1              # Move to next respondent or to the summary
    st.session_state.page_index   = (2 if st.session_state.current_idx < st.session_state.num_respondents else 99) # show global summary page
                                    

################################################################################
#  Summary page                                                                #
################################################################################
    
def final_summary():
    """Render a human-readable summary of all respondents and save a combined JSON snapshot."""

    st.title("✅  All respondents complete!")

    if not st.session_state.survey_data:
        st.info("No data collected yet.")
        return

    # ------------------------------------------------------------------ table -
    for rec in st.session_state.survey_data:
        rid = rec.get("id", "—")
        st.markdown(f"### Respondent ID: **{rid}**")

        methods_dict = rec.get("Methods") or {}
        if not methods_dict:
            st.warning("No method blocks recorded for this respondent.")
            continue

        for meth, block in methods_dict.items():
            st.markdown(f"**Method:** {meth}")
            util_dict = block.get("utility", {})
            if not util_dict:
                st.write("→ No utilities recorded.")
                continue

            # pretty print utilities sorted high → low
            ordered = sorted(dev_load_map,
                             key=lambda d: util_dict.get(d, 0.0),
                             reverse=True)
            for d in ordered:
                st.write(f"• {d}: {util_dict.get(d, 0.0):.1f}")

        st.markdown("---")

    # ---------------------------------------------------------------- export -
    # Optional: save one combined JSON so you can reload outside Streamlit
    with open("all_respondents.json", "w") as f:
        json.dump(st.session_state.survey_data, f, indent=2)
    st.success("⤵︎  All data saved to *all_respondents.json* in the app folder.")

# -------------------- Helpers for folders and for optimization --------------------------------

OUTDIR = Path("outputs")
OUTDIR.mkdir(exist_ok=True)

def save_chart(chart: alt.Chart, stem: str):
    """Export an Altair chart to PNG/SVG using the vl-convert engine."""
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
# Relaxed KP (HiGHS) + exact KP to produce two bundles and convenience totals.

def run_optimisation(util_dict, power_map, P):
    """Build a dataframe and compute both LP-relaxation (HiGHS via SciPy) and exact DP bundles; also totals for power/utility."""
    #Return a dataframe with LP & DP selections and some totals.
    
    df = pd.DataFrame({
        "Device": list(util_dict),
        "Utility": [util_dict[d]  for d in util_dict],
        "Power":   [power_map[d] for d in util_dict]
    })
    df["Utility_per_Watt"] = df["Utility"] / df["Power"] 
    # ----------------------------- relaxed KP ----------------------------------------------
    c = -df["Utility"].to_numpy()
    res = linprog(c,
                  A_ub=[df["Power"].to_numpy()],
                  b_ub=[P],
                  bounds=[(0,1)]*len(df),
                  method="highs")
    df["LP_pick"] = np.round(res.x).astype(int)

    # ------------------------------ exact KP ----------------------------------------------
    df["DP_pick"] = knapsack_dp(df["Power"].tolist(),
                                 df["Utility"].tolist(), P)

    # totals for convenience
    for col in ("LP_pick","DP_pick"):
        df[f"{col}_power"]   = df["Power"]   * df[col]
        df[f"{col}_utility"] = df["Utility"] * df[col]
    return df

# --------------------------- Analytics --------------------------------------
    
def analytics_page():
    """ Aggregate respondents’ utilities, compare methods (SG/PC/ES), visualise summaries, and run the optimisation dashboard."""
    st.title("📊 Survey analytics")

    final_summary()
    
    # ─────────────────── reshape raw survey_data ────────────────────────────
    # rows: respondent × method × device  → utility
    rows = []
    for rec in st.session_state.survey_data:
        rid = rec["id"]
        for method, block in rec["Methods"].items():
            for dev, util in block["utility"].items():
                rows.append({"Respondent": rid,
                             "Method": method,
                             "Device": dev,
                             "Utility": util})
    df = pd.DataFrame(rows)

    if df.empty:
        st.info("No data yet – finish at least one respondent first.")
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

    st.subheader("How often is each device ranked #1?")
    st.dataframe(top1_counts.to_frame())   # tabular view
    st.altair_chart(
        alt.Chart(top1_counts.reset_index(),
                  title="Frequency of being ranked #1").mark_bar().encode(
            x="Top-1 count:Q",
            y=alt.Y("Device:N", sort="-x")
        ), use_container_width=True
    )

    #---------------------------- mean utilities ------------------------------------------
    mean_util = (
        df.groupby("Device")["Utility"]
        .mean()
        .rename("Average utility")
        .reindex(dev_load_map)
    )

    st.subheader("Average utility per device (0-100 %)")
    st.dataframe(mean_util.to_frame())
    st.altair_chart(
        alt.Chart(mean_util.reset_index(),
                  title="Mean utility").mark_bar().encode(
            x="Average utility:Q",
            y=alt.Y("Device:N", sort="-x")
        ), use_container_width=True
    )

    # ----------------------------- per-method breakdown -------------------------------
    st.header("Method comparison")
    
    # -------------------------- 1. plain-text winners ------------------------------------
    winners = {}                         
    for meth in ("SG", "PC", "ES"):
        meth_df = df[df["Method"] == meth]
        if meth_df.empty:
            continue
        winners[meth] = (
            meth_df.groupby("Respondent")["Utility"].idxmax()
                   .map(df.loc[:, "Device"])
                   .value_counts()
                   .idxmax()
        )
    
    overall_winner = (
        df.loc[df.groupby(["Respondent", "Method"])["Utility"].idxmax(), "Device"]
          .value_counts()
          .idxmax()
    )
    
    st.markdown(
        "* **Overall #1 device:** {ow}\n"
        "* **SG #1 device:** {sg}\n"
        "* **PC #1 device:** {pc}\n"
        "* **ES #1 device:** {es}".format(
            ow=overall_winner,
            sg=winners.get("SG", "–"),
            pc=winners.get("PC", "–"),
            es=winners.get("ES", "–")
        )
    )
    # --------------------- 2. combined utility bar-chart ----------------------------
    # prepare long form dataframe with a colour label
    combo = (
        df.groupby(["Method", "Device"])["Utility"]
          .mean()
          .reset_index()
    )
    
    method_palette = {"SG": "#1f77b4",   # blue
                      "PC": "#d62728",   # red
                      "ES": "#2ca02c"}   # green
    
    bars = []
    for meth in ("SG", "PC", "ES"):
        if meth not in combo.Method.unique():
            continue
        bars.append(
            alt.Chart(combo[combo.Method == meth])
                .mark_bar(color=method_palette[meth])
                .encode(
                    x=alt.X("Utility:Q", scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y("Device:N", sort=dev_load_map)
                )
                .properties(title=f"{meth} mean")
        )
    
    st.altair_chart(alt.hconcat(*bars), use_container_width=True)
    
    # ------------------------- slope chart --------------------------------
    def slope_chart(df_long, meth_left, meth_right,
        """See inline comments. Docstring placeholder."""
                    symbol_left="circle", symbol_right="circle",
                    title=""):
        tbl = (
            df_long[df_long.Method.isin([meth_left, meth_right])]
              .groupby(["Method", "Device"])["Utility"].mean()
              .unstack("Method")
              .reindex(dev_load_map)
              .reset_index()
        )
    
        base = alt.Chart(tbl).encode(
            y=alt.Y("Device:N", sort=dev_load_map, title=None),
            color="Device:N"
        )
    
        lines = base.mark_line().encode(
            x=alt.X(f"{meth_left}:Q", scale=alt.Scale(domain=[0, 100]),
                    axis=alt.Axis(title="Utility (%)")),
            x2=f"{meth_right}:Q"
        )
    
        if meth_left == "SG" or meth_left == "ES":
            pts_left = base.mark_point(filled=True, size=80, shape=symbol_left)\
                            .encode(x=f"{meth_left}:Q")
        elif meth_left == "PC":
            pts_left = base.mark_point(filled=False, size=80, shape=symbol_left)\
                            .encode(x=f"{meth_left}:Q")

        if meth_right == "SG" or meth_right == "ES":       
            pts_right= base.mark_point(filled=True, size=80, shape=symbol_right)\
                            .encode(x=f"{meth_right}:Q")
        elif meth_right == "PC":
            pts_right= base.mark_point(filled=False, size=80, shape=symbol_right)\
                            .encode(x=f"{meth_right}:Q")
            
        return (lines + pts_left + pts_right).properties(
            title=title, width=380
        )    

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
    
    st.subheader("Mean utility – pairwise comparison")
    
    chart_sg_pc = slope_chart(df, "SG", "PC",
                              symbol_left="circle", symbol_right="circle",
                              title="SG (●)  →  PC (○)")
    chart_sg_es = slope_chart(df, "SG", "ES",
                              symbol_left="circle", symbol_right="triangle-up",
                              title="SG (●)  →  ES (▲)")
    chart_pc_es = slope_chart(df, "PC", "ES",
                              symbol_left="circle", symbol_right="triangle-up",
                              title="PC (○)  →  ES (▲)")
    
    st.altair_chart(
        alt.hconcat(chart_sg_pc, chart_sg_es, chart_pc_es),
        use_container_width=True
    )

    # ------------------- crossover ranking chart -----------------------------
    # build a “long” table: one row per device × side
    def crossover_plot(rank_left, rank_right, title):
        cross = pd.DataFrame(
            [{"Device": d, "Side": rank_left.name,  "x": 0, "rank": rank_left[d]}
             for d in dev_load_map] +
            [{"Device": d, "Side": rank_right.name, "x": 1, "rank": rank_right[d]}
             for d in dev_load_map]
        )
    
        base = alt.Chart(cross).encode(
            x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[-0.1, 1.1])),
            y=alt.Y("rank:Q",
                    axis=None,
                    scale=alt.Scale(domain=[0.5, len(dev_load_map) + .5],
                                    reverse=True)),
            color="Device:N",
            detail="Device:N"
        )
    
        chart = (
            # connecting lines
            base.mark_line(strokeWidth=1.3) +
    
            # points on the left
            base.transform_filter(alt.datum.Side == rank_left.name)
                .mark_point(filled=True, size=80, shape="circle") +
    
            # points on the right
            base.transform_filter(alt.datum.Side == rank_right.name)
                .mark_point(filled=True, size=80, shape="triangle-up") +
    
            # labels, with conditional offset & alignment
            base.mark_text(fontSize=9).encode(
                text="Device:N",
                dx=alt.condition(
                    alt.datum.Side == rank_left.name,
                    alt.value(-8),      # when on the left
                    alt.value(8)        # when on the right
                ),
                align=alt.condition(
                    alt.datum.Side == rank_left.name,
                    alt.value("right"),
                    alt.value("left")
                )
            )
        ).properties(
            title=title,
            width=250,
            height=22 * len(dev_load_map)
        ).configure_view(stroke=None)
    
        return chart

    # --------------------- energy-budget optimisation -------------------------
    st.header("Optimised device bundle")
    
    if st.session_state.max_power is None:
        st.info("Maximum power not set – configure it on the setup page.")
        return
    
    # Build one utility number per device according to the survey-taker’s choice
    choice = st.session_state.utility_source   # "PC", "SG", "ES", "Average"

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

################################################################################
#  Main Menu                                                                   #
################################################################################
    
def main():
    """Top-level router based on st.session_state.page_index."""
    page = st.session_state.page_index
    if page == 0: 
        survey_setup_page()
    elif page == 1: 
        device_availability_page()   
    elif page == 2: 
        respondent_intro_page()
    elif page == 5:
        run_selected_method()
    elif page == 6:
        respondent_method_page()
    else: 
        analytics_page()

if __name__ == "__main__":
    main()
