import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = ["lepton_pT", "lepton_eta",
            "lepton phi", "missing_energy_magnitude",
            "missing_energy_phi", "jet_1_pt", "jet_1_eta",
            "jet_1_phi", "jet_1_btag", "jet_2_pt", "jet_2_eta",
            "jet_2_phi", "jet_2_btag", "jet_3_pt",
            "jet_3_eta", "jet_3_phi", "jet_3_btag",
            "jet_4_pt", "jet_4_eta", "jet_4_phi",
            "jet_4_btag", "m_jj", "m_jjj", "m_lv",
            "m_jlv", "m_bb", "m_wbb", "m_wwbb"]


def load_uci_higgs(file_name, train_test_split_seed=None):
    data_full = pd.read_csv(file_name, names=["label"] + FEATURES, header=None)
    train, test = train_test_split(data_full, test_size=0.2, shuffle=True, random_state=train_test_split_seed)
    return train, test 
