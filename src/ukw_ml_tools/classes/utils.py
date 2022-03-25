def split_size_str(size_str):
    if size_str == "unknown":
        result = None
    elif size_str == "<5":
        result = (0, 4.99)
    elif size_str == "5-10":
        result = (5,10)
    elif size_str == ">10-20":
        result = (10.01, 20)
    elif size_str == ">20":
        result = (20.01, 999)

    return result

_ = {
    "rounds": [
        "2nd_round",
        "4th_round",
        "5th_round",
        "6th_round"
    ],
    "Boeck": [
        "Boeck-2",
        "Boeck1",
        "Boeck2",
        "BoeckRetro",
        "boeck2"
    ],
    "Heil": [
        "Heil_1",
        "Heil_2",
        "heil"
    ],
    "Heubach": [
        "HeubachEndoBox",
        "HeubachEndoBox (Kopie)",
        "heubach"
    ],
    "UKW": [
        "Koloskopie"
    ],
    "Ludwig": [
        "Ludwig",
        "Ludwig1"
    ],
    "Passek": [
        "PassekEndoBox",
        "PassekRecordingPC",
        "passek"
    ],
    "Simonis": [
        "Simonis_1",
        "Simonis_2",
        "simonis"
    ],
    "Stuttgart": [
        "Stuttgart"
    ],
    "Stuttgart Archiv": ["archive_stuttgart"],
    "Würzburg Archiv": ["archive_würzburg"],
    "GI Genius Retrospective": ["gi_genius_retrospective", "gi_genius_retrospective_frames", "gi_genius_ulm"]

}



ORIGIN_LOOKUP = {}
for key, value in _.items():
    for _ in value:
        ORIGIN_LOOKUP[_.lower()] = key


POLYP_EVALUATION_STRUCTURE = {
        "location_segment": {
        "attribute": "location_segment",
        "required": True,
        "required_if": []
        },
        "size_category": {
        "attribute": "size_category",
        "required": True,
        "required_if": []
        },
        "surface_intact": {
        "attribute": "surface_intact",
        "required": True,
        "required_if": [],
        },
        "rating": {
        "attribute": "rating",
        "required": True,
        "required_if": [],
        },
        "resection": {
        "attribute": "resection",
        "required": True,
        "required_if": [],
        },
        "location_cm": {
        "attribute": "location_cm",
        "required": False,
        "required_if": [
            {
            "attribute": "location_segment",
            "values": [
                "rectum",
                "sigma"
            ]
            }
        ],
        },
        "size_mm": {
        "attribute": "polyp_size_mm",
        "required": False,
        "required_if": []
        },
        "paris": {
        "attribute": "polyp_paris",
        "required": False,
        "required_if": [
            {
            "attribute": "size_category",
            "values": [
                "<5",
                "5-10",
                ">10-20",
                ">20"
            ]
            }
        ]
        },
        "lst": {
        "attribute": "lst",
        "required": False,
        "required_if": [
            {
            "attribute": "size_category",
            "values": [
                ">10-20",
                ">20"
            ]
            }
        ],
        },
        "nice": {
        "attribute": "nice",
        "required": False,
        "required_if": [
            {
            "attribute": "size_category",
            "values": [
                "5-10",
                ">10-20",
                ">20"
            ]
            }
        ],
        },
        "non_lifting_sign": {
        "attribute": "non_lifting_sign",
        "required": False,
        "required_if": [],
        },
        "tool": {
        "attribute": "tool",
        "required": False,
        "required_if": [
            {
            "attribute": "resection",
            "values": [
                True
            ]
            }
        ],
        },
        # "resection_technique": {
        # "attribute": "resection_technique",
        # "required": False,
        # "required_if": [
        #     {
        #     "attribute": "resection",
        #     "values": [
        #         True
        #     ]
        #     }
        # ],
        # },
        "salvage": {
        "attribute": "salvage",
        "required": False,
        "required_if": [
            {
            "attribute": "resection",
            "values": [
                True
            ]
            }
        ],
        },
        "ectomy_wound_care": {
        "attribute": "ectomy_wound_care",
        "required": False,
        "required_if": [
            {
            "attribute": "resection",
            "values": [
                True
            ]
            }
        ],
        },
        "ectomy_wound_care_technique": {
        "attribute": "ectomy_wound_care_technique",
        "required": False,
        "required_if": [
            {
            "attribute": "ectomy_wound_care",
            "values": [
                True
            ]
            }
        ],
        },
        "apc_watts": {
        "attribute": "apc_watts",
        "required": False,
        "required_if": [
            {
            "attribute": "ectomy_wound_care_technique",
            "values": [
                "apc"
            ]
            }
        ],
        },
        "number_clips": {
        "attribute": "number_clips",
        "required": False,
        "required_if": [
            {
            "attribute": "ectomy_wound_care_technique",
            "values": [
                "clip"
            ]
            }
        ],
        },
        "ectomy_wound_care_success": {
        "attribute": "ectomy_wound_care_success",
        "required": False,
        "required_if": [
            {
            "attribute": "ectomy_wound_care",
            "values": [
                True
            ]
            }
        ],
        },
        "no_resection_reason": {
        "attribute": "no_resection_reason",
        "required": False,
        "required_if": [
            {
            "attribute": "resection",
            "values": [
                False
            ]
            }
        ],
        },
        "resection_technique": {
        "attribute": "resection_technique",
        "required": False,
        "required_if": [
            {
            "attribute": "resection",
            "values": [
                True
            ]
            }
        ]
        }
        }
        