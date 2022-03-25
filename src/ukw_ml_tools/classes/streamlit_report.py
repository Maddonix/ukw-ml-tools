import streamlit as st

class ReportAnnotation:
    def __init__(self, init_streamlit = True):
        self.sedation_options = ["unknown", "no", "propofol", "midazolam", "propofol+midazolam", "other"]
        self.location_options = [
            "unknown",
            "rectum",
            "sigma",
            "descendens",
            "transversum",
            "ascendens",
            "caecum",
            "terminal_ileum",
            "neoterminal_ileum",
            "right_flexure",
            "left_flexure",
            "right_colon",
            "left_colon"
        ]
        self.size_options = [
            "unknown",
            "<5",
            "5-10",
            ">10-20",
            ">20"
        ]
        self.rating_options = [
            "unknown",
            "hyperplastic",
            "adenoma",
            "ssa",
            "inflammatory",
            "dalm",
            "malign"
        ]
        self.paris_options = [
            "unknown",
            "Is",
            "Ip",
            "Ix",
            "IIa",
            "IIb",
            "IIc",
            "IIx"
        ]        
        self.morphology_options = [
            "unknown",
            "sessil",
            "flach",
            "gestielt"
        ]
        self.histo_options = [
            "unknown", "non_adenoma", "tubular_adenoma", "tubulovillous_adenoma", "sessil_serrated_lesion", "carcinoma", "not_evaluated"
        ]
        self.dysplasia_options = [
            "unknown", "no", "low", "high"
        ]
        self.tool_options = ["unknown", "grasper", "sling_hot", "sling_cold", "sling"]
        self.wound_care_success_options = ["unknown", "preventive", "hemostasis", "no_hemostasis", "reactivation_hemostasis", "reactivation_no_hemostasis"]

        self.expander_general = st.empty()
        self.expander_polyps_list = []

        if init_streamlit:
            self.results = self.get_base_inputs()
            self.get_polyp_inputs_base()

    def clear_inputs(self):
        self.expander_general.empty()
        for _ in self.expander_polyps_list:
            _.empty()
        self.results = self.get_base_inputs()
        self.get_polyp_inputs_base()

    def get_base_inputs(self):  
        self.expander_general = st.expander("General", True)
        with self.expander_general:
            cols = st.columns(6)
            results = {
                "intervention_time": cols[0].number_input("Intervention Time (s)", -1, step = 1, key = "intervention_time"),
                "withdrawal_time": cols[1].number_input("Withdrawal Time (s)", -1, step = 1),
                "sedation": cols[2].selectbox("Sedation", self.sedation_options),
                "bbps_worst": cols[3].number_input("BBPS Worst", -1, 3, -1, step = 1),
                "bbps_total": cols[4].number_input("BBPS total", -1, 9, -1, step = 1),
                "n_polyps": int(cols[5].number_input("Number of Polyps", 0, step = 1)),
                "other_pathologies": cols[1].checkbox("Other Pathologies"),
                "indication": cols[0].selectbox("Indication", ["unknown", "screening", "surveillance", "symptomatic"]),
                "mark_other": cols[2].checkbox("Mark Other"),
                "polyps": {}
            }  

        return results

    def get_polyp_inputs_base(self):
        self.expander_polyps_list = [st.expander(f"Polyp {i}", False) for i in range(0, self.results["n_polyps"])]
        for i in range(0, self.results["n_polyps"]):
            with self.expander_polyps_list[i]:
                cols = st.columns(6)
                _polyp_inputs = {
                    "location_segment": cols[0].selectbox("Location", self.location_options, key = f"polyp_{i}_location_segment"),
                    "location_cm": cols[0].number_input("Location cm", -1, value = -1, step = 1, key = f"polyp_{i}_location_cm"),
                    "size_category": cols[0].selectbox("Size", self.size_options, key = f"polyp_{i}_size_category"),
                    "size_mm": cols[0].number_input("Size mm", -1, value = -1, step = 1, key = f"polyp_{i}_size_mm"),
                    "surface_intact": cols[1].selectbox("Surface intact", ["unknown", False, True], key = f"polyp_{i}_surface"),
                    "rating": cols[2].selectbox("Rating", self.rating_options, key = f"polyp_{i}_rating"),
                    "morphology": cols[1].selectbox("Morphology", self.morphology_options, key = f"polyp_{i}_morphology"),
                    "paris": cols[1].multiselect("Paris", self.paris_options, key = f"polyp_{i}_paris"),
                    "nice": cols[1].selectbox("NICE", ["unknown","I","II","III"], key = f"polyp_{i}_nice"),
                    "lst": cols[2].selectbox("LST", ["unknown", "granular", "non_granular", "mixed"], key=f"polyp_{i}_lst"),
                    "dysplasia": cols[2].selectbox("Dysplasia", self.dysplasia_options, key = f"polyp_{i}_dysplasia"),
                    "histo": cols[2].selectbox("Histology", self.histo_options, key = f"polyp_{i}_histo"),
                    "resection": cols[3].selectbox("Resection", ["unknown", False, True], key = f"polyp_{i}_resection", index = 1),
                    "injection": cols[3].selectbox("Injection", ["unknown", False, True], key = f"polyp_{i}_injection"),
                }
                if _polyp_inputs["injection"]:
                    _polyp_inputs["non_lifting_sign"] = cols[3].selectbox("Non Lifting Sign", ["unknown", False, True], key = f"polyp_{i}_non_lifting")
                if _polyp_inputs["resection"] == True:
                    _polyp_inputs["tool"] = cols[4].selectbox("Resection Tool", self.tool_options, key = f"polyp{i}_tool")
                    _polyp_inputs["resection_technique"] = cols[4].selectbox("Resection Technique", ["unknown", "enbloc", "piecemeal", "incomplete"], key = f"polyp_{i}_resection_technique")
                    _polyp_inputs["salvage"] = cols[4].selectbox("Salvage", ["unknown", False, True], key = f"polyp_{i}_salvage")
                    _polyp_inputs["ectomy_wound_care"] = cols[4].selectbox("Wound Care", ["unknown", False, True], key = f"polyp_{i}_wound_care")
                    _polyp_inputs["resection_status_microscopic"] = cols[4].selectbox("Resection Status Microscopic", ["unknown", "R0", "R1", "R2"], key = f"polyp_{i}_resection_microscopic")
                
                    if _polyp_inputs["ectomy_wound_care"] == True:
                        _polyp_inputs["ectomy_wound_care_technique"] = cols[5].selectbox("Wound Care Technique", ["unknown", "clip", "apc"], key = f"polyp_{i}_wound_care_technique")
                        _polyp_inputs["ectomy_wound_care_success"] = cols[5].selectbox("Wound Care Success", self.wound_care_success_options, key = f"polyp_{i}_wound_care_success")

                        if _polyp_inputs["ectomy_wound_care_technique"] == "apc":
                            _polyp_inputs["apc_watts"] = cols[5].number_input("APC Watts", -1, step = 1, key = f"polyp_{i}_apc_watts")
                        elif _polyp_inputs["ectomy_wound_care_technique"] == "clip":
                            _polyp_inputs["number_clips"] = cols[5].number_input("Number Clips", -1, step = 1, key = f"polyp_{i}_number_clips")

                else:
                    _polyp_inputs["no_resection_reason"] = cols[4].selectbox("Reason for no Resection", ["unknown", "provided"], key = f"polyp_{i}_no_resection_reason")             

            self.results["polyps"][i] = _polyp_inputs
