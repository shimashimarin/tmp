import uproot as up
import awkward as ak
import numpy as np
import coffea
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea import processor
import hist
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path
import time


def get_file_list(fpath, pattern):
    flist = []
    for path in Path(fpath).rglob(pattern):
        # print(path.resolve())
        flist.append(str(path.resolve()))
    return flist


class WJetProcessor(processor.ProcessorABC):
    def __init__(self):
        self.trigger_group = ".*SingleEle.*"
        self.analysis = "tagAndProbe"

        self.prefixes = {"tag": "tag", "probe": "probe"}

    def process(self, events: ak.Array):
        dataset = events.metadata["dataset"]

        genlep_pt_min = 20
        genjet_pt_min = 20
        lep_jet_dr = 0.4

        count_dict = defaultdict(int)
        count_dict["ntot"] += len(events)
        count_dict["npos"] += ak.sum(events.genWeight > 0)
        count_dict["nneg"] += ak.sum(events.genWeight < 0)
        count_dict["neff"] += ak.sum(events.genWeight > 0) - ak.sum(
            events.genWeight < 0
        )

        nbins = 100

        dataset_axis = hist.axis.StrCategory(
            [], growth=True, name="dataset", label="Primary dataset"
        )
        nlep_axis = hist.axis.Regular(4, 0, 4, name="x", label="nLepton")
        lep1_pt_axis = hist.axis.Regular(
            nbins, genlep_pt_min, 150, name="x", label=r"$p_{T}~(\ell 1)$ [GeV]"
        )
        lep_eta_axis = hist.axis.Regular(nbins, -2.5, 2.5, name="x", label=r"$\eta$")
        lep_phi_axis = hist.axis.Regular(
            nbins, -np.pi, np.pi, name="x", label=r"$\phi$"
        )

        njet_axis = hist.axis.Regular(6, 0, 6, name="x", label="nLepton")
        jet1_pt_axis = hist.axis.Regular(
            nbins, genjet_pt_min, 150, name="x", label=r"$p_{T}~(j 1)$ [GeV]"
        )
        jet1_mass_axis = hist.axis.Regular(
            nbins, 0, 40, name="x", label=r"$mass~(j 1)$ [GeV]"
        )
        jet2_pt_axis = hist.axis.Regular(
            nbins, genjet_pt_min, 100, name="x", label=r"$p_{T}~(j 2)$ [GeV]"
        )
        jet2_mass_axis = hist.axis.Regular(
            nbins, 0, 36, name="x", label=r"$mass~(j 2)$ [GeV]"
        )
        jet3_pt_axis = hist.axis.Regular(
            int(nbins * 0.5), genjet_pt_min, 100, name="x", label=r"$p_{T}~(j 3)$ [GeV]"
        )
        jet3_mass_axis = hist.axis.Regular(
            int(nbins * 0.5), 0, 25, name="x", label=r"$mass~(j 3)$ [GeV]"
        )
        jet4_pt_axis = hist.axis.Regular(
            int(nbins * 0.3), genjet_pt_min, 80, name="x", label=r"$p_{T}~(j 4)$ [GeV]"
        )
        jet4_mass_axis = hist.axis.Regular(
            int(nbins * 0.3), 0, 20, name="x", label=r"$mass~(j 4)$ [GeV]"
        )
        jet_eta_axis = hist.axis.Regular(nbins, -2.5, 2.5, name="x", label=r"$\eta$")
        jet_phi_axis = hist.axis.Regular(
            nbins, -np.pi, np.pi, name="x", label=r"$\phi$"
        )
        met_pt_axis = hist.axis.Regular(
            nbins, 0, 100, name="x", label=r"$p_{T}^{miss}$ [GeV]"
        )
        met_phi_axis = hist.axis.Regular(
            nbins, -np.pi, np.pi, name="x", label=r"$\phi$"
        )

        h_lhe_nlep = hist.Hist(
            dataset_axis, nlep_axis, storage="weight", label="Counts"
        )
        h_lhe_e1_pt = hist.Hist(
            dataset_axis, lep1_pt_axis, storage="weight", label="Counts"
        )
        h_lhe_e1_eta = hist.Hist(
            dataset_axis, lep_eta_axis, storage="weight", label="Counts"
        )
        h_lhe_e1_phi = hist.Hist(
            dataset_axis, lep_phi_axis, storage="weight", label="Counts"
        )
        h_lhe_mu1_pt = hist.Hist(
            dataset_axis, lep1_pt_axis, storage="weight", label="Counts"
        )
        h_lhe_mu1_eta = hist.Hist(
            dataset_axis, lep_eta_axis, storage="weight", label="Counts"
        )
        h_lhe_mu1_phi = hist.Hist(
            dataset_axis, lep_phi_axis, storage="weight", label="Counts"
        )
        h_lhe_l1_pt = hist.Hist(
            dataset_axis, lep1_pt_axis, storage="weight", label="Counts"
        )
        h_lhe_l1_eta = hist.Hist(
            dataset_axis, lep_eta_axis, storage="weight", label="Counts"
        )
        h_lhe_l1_phi = hist.Hist(
            dataset_axis, lep_phi_axis, storage="weight", label="Counts"
        )
        h_lhe_njet = hist.Hist(
            dataset_axis, njet_axis, storage="weight", label="Counts"
        )
        h_lhe_j1_pt = hist.Hist(
            dataset_axis, jet1_pt_axis, storage="weight", label="Counts"
        )
        h_lhe_j2_pt = hist.Hist(
            dataset_axis, jet2_pt_axis, storage="weight", label="Counts"
        )
        h_lhe_j3_pt = hist.Hist(
            dataset_axis, jet3_pt_axis, storage="weight", label="Counts"
        )
        h_lhe_j4_pt = hist.Hist(
            dataset_axis, jet4_pt_axis, storage="weight", label="Counts"
        )
        h_lhe_j1_eta = hist.Hist(
            dataset_axis, jet_eta_axis, storage="weight", label="Counts"
        )
        h_lhe_j2_eta = hist.Hist(
            dataset_axis, jet_eta_axis, storage="weight", label="Counts"
        )
        h_lhe_j3_eta = hist.Hist(
            dataset_axis, jet_eta_axis, storage="weight", label="Counts"
        )
        h_lhe_j4_eta = hist.Hist(
            dataset_axis, jet_eta_axis, storage="weight", label="Counts"
        )
        h_lhe_j1_phi = hist.Hist(
            dataset_axis, jet_phi_axis, storage="weight", label="Counts"
        )
        h_lhe_j2_phi = hist.Hist(
            dataset_axis, jet_phi_axis, storage="weight", label="Counts"
        )
        h_lhe_j3_phi = hist.Hist(
            dataset_axis, jet_phi_axis, storage="weight", label="Counts"
        )
        h_lhe_j4_phi = hist.Hist(
            dataset_axis, jet_phi_axis, storage="weight", label="Counts"
        )
        h_lhe_j1_mass = hist.Hist(
            dataset_axis, jet1_mass_axis, storage="weight", label="Counts"
        )
        h_lhe_j2_mass = hist.Hist(
            dataset_axis, jet2_mass_axis, storage="weight", label="Counts"
        )
        h_lhe_j3_mass = hist.Hist(
            dataset_axis, jet3_mass_axis, storage="weight", label="Counts"
        )
        h_lhe_j4_mass = hist.Hist(
            dataset_axis, jet4_mass_axis, storage="weight", label="Counts"
        )
        # h_lhe_met_pt = hist.Hist(dataset_axis, met_pt_axis, storage="weight", label="Counts")
        # h_lhe_met_phi = hist.Hist(dataset_axis, met_phi_axis, storage="weight", label="Counts")

        h_gen_nlep = hist.Hist(
            dataset_axis, nlep_axis, storage="weight", label="Counts"
        )
        h_gen_e1_pt = hist.Hist(
            dataset_axis, lep1_pt_axis, storage="weight", label="Counts"
        )
        h_gen_e1_eta = hist.Hist(
            dataset_axis, lep_eta_axis, storage="weight", label="Counts"
        )
        h_gen_e1_phi = hist.Hist(
            dataset_axis, lep_phi_axis, storage="weight", label="Counts"
        )
        h_gen_mu1_pt = hist.Hist(
            dataset_axis, lep1_pt_axis, storage="weight", label="Counts"
        )
        h_gen_mu1_eta = hist.Hist(
            dataset_axis, lep_eta_axis, storage="weight", label="Counts"
        )
        h_gen_mu1_phi = hist.Hist(
            dataset_axis, lep_phi_axis, storage="weight", label="Counts"
        )
        h_gen_l1_pt = hist.Hist(
            dataset_axis, lep1_pt_axis, storage="weight", label="Counts"
        )
        h_gen_l1_eta = hist.Hist(
            dataset_axis, lep_eta_axis, storage="weight", label="Counts"
        )
        h_gen_l1_phi = hist.Hist(
            dataset_axis, lep_phi_axis, storage="weight", label="Counts"
        )
        h_gen_njet = hist.Hist(
            dataset_axis, njet_axis, storage="weight", label="Counts"
        )
        h_gen_j1_pt = hist.Hist(
            dataset_axis, jet1_pt_axis, storage="weight", label="Counts"
        )
        h_gen_j2_pt = hist.Hist(
            dataset_axis, jet2_pt_axis, storage="weight", label="Counts"
        )
        h_gen_j3_pt = hist.Hist(
            dataset_axis, jet3_pt_axis, storage="weight", label="Counts"
        )
        h_gen_j4_pt = hist.Hist(
            dataset_axis, jet4_pt_axis, storage="weight", label="Counts"
        )
        h_gen_j1_eta = hist.Hist(
            dataset_axis, jet_eta_axis, storage="weight", label="Counts"
        )
        h_gen_j2_eta = hist.Hist(
            dataset_axis, jet_eta_axis, storage="weight", label="Counts"
        )
        h_gen_j3_eta = hist.Hist(
            dataset_axis, jet_eta_axis, storage="weight", label="Counts"
        )
        h_gen_j4_eta = hist.Hist(
            dataset_axis, jet_eta_axis, storage="weight", label="Counts"
        )
        h_gen_j1_phi = hist.Hist(
            dataset_axis, jet_phi_axis, storage="weight", label="Counts"
        )
        h_gen_j2_phi = hist.Hist(
            dataset_axis, jet_phi_axis, storage="weight", label="Counts"
        )
        h_gen_j3_phi = hist.Hist(
            dataset_axis, jet_phi_axis, storage="weight", label="Counts"
        )
        h_gen_j4_phi = hist.Hist(
            dataset_axis, jet_phi_axis, storage="weight", label="Counts"
        )
        h_gen_j1_mass = hist.Hist(
            dataset_axis, jet1_mass_axis, storage="weight", label="Counts"
        )
        h_gen_j2_mass = hist.Hist(
            dataset_axis, jet2_mass_axis, storage="weight", label="Counts"
        )
        h_gen_j3_mass = hist.Hist(
            dataset_axis, jet3_mass_axis, storage="weight", label="Counts"
        )
        h_gen_j4_mass = hist.Hist(
            dataset_axis, jet4_mass_axis, storage="weight", label="Counts"
        )
        h_gen_met_pt = hist.Hist(
            dataset_axis, met_pt_axis, storage="weight", label="Counts"
        )
        h_gen_met_fid_pt = hist.Hist(
            dataset_axis, met_pt_axis, storage="weight", label="Counts"
        )
        h_gen_met_phi = hist.Hist(
            dataset_axis, met_phi_axis, storage="weight", label="Counts"
        )

        h_gencut_nlep = hist.Hist(
            dataset_axis, nlep_axis, storage="weight", label="Counts"
        )
        h_gencut_e1_pt = hist.Hist(
            dataset_axis, lep1_pt_axis, storage="weight", label="Counts"
        )
        h_gencut_e1_eta = hist.Hist(
            dataset_axis, lep_eta_axis, storage="weight", label="Counts"
        )
        h_gencut_e1_phi = hist.Hist(
            dataset_axis, lep_phi_axis, storage="weight", label="Counts"
        )
        h_gencut_mu1_pt = hist.Hist(
            dataset_axis, lep1_pt_axis, storage="weight", label="Counts"
        )
        h_gencut_mu1_eta = hist.Hist(
            dataset_axis, lep_eta_axis, storage="weight", label="Counts"
        )
        h_gencut_mu1_phi = hist.Hist(
            dataset_axis, lep_phi_axis, storage="weight", label="Counts"
        )
        h_gencut_l1_pt = hist.Hist(
            dataset_axis, lep1_pt_axis, storage="weight", label="Counts"
        )
        h_gencut_l1_eta = hist.Hist(
            dataset_axis, lep_eta_axis, storage="weight", label="Counts"
        )
        h_gencut_l1_phi = hist.Hist(
            dataset_axis, lep_phi_axis, storage="weight", label="Counts"
        )
        h_gencut_njet = hist.Hist(
            dataset_axis, njet_axis, storage="weight", label="Counts"
        )
        h_gencut_j1_pt = hist.Hist(
            dataset_axis, jet1_pt_axis, storage="weight", label="Counts"
        )
        h_gencut_j2_pt = hist.Hist(
            dataset_axis, jet2_pt_axis, storage="weight", label="Counts"
        )
        h_gencut_j3_pt = hist.Hist(
            dataset_axis, jet3_pt_axis, storage="weight", label="Counts"
        )
        h_gencut_j4_pt = hist.Hist(
            dataset_axis, jet4_pt_axis, storage="weight", label="Counts"
        )
        h_gencut_j1_eta = hist.Hist(
            dataset_axis, jet_eta_axis, storage="weight", label="Counts"
        )
        h_gencut_j2_eta = hist.Hist(
            dataset_axis, jet_eta_axis, storage="weight", label="Counts"
        )
        h_gencut_j3_eta = hist.Hist(
            dataset_axis, jet_eta_axis, storage="weight", label="Counts"
        )
        h_gencut_j4_eta = hist.Hist(
            dataset_axis, jet_eta_axis, storage="weight", label="Counts"
        )
        h_gencut_j1_phi = hist.Hist(
            dataset_axis, jet_phi_axis, storage="weight", label="Counts"
        )
        h_gencut_j2_phi = hist.Hist(
            dataset_axis, jet_phi_axis, storage="weight", label="Counts"
        )
        h_gencut_j3_phi = hist.Hist(
            dataset_axis, jet_phi_axis, storage="weight", label="Counts"
        )
        h_gencut_j4_phi = hist.Hist(
            dataset_axis, jet_phi_axis, storage="weight", label="Counts"
        )
        h_gencut_j1_mass = hist.Hist(
            dataset_axis, jet1_mass_axis, storage="weight", label="Counts"
        )
        h_gencut_j2_mass = hist.Hist(
            dataset_axis, jet2_mass_axis, storage="weight", label="Counts"
        )
        h_gencut_j3_mass = hist.Hist(
            dataset_axis, jet3_mass_axis, storage="weight", label="Counts"
        )
        h_gencut_j4_mass = hist.Hist(
            dataset_axis, jet4_mass_axis, storage="weight", label="Counts"
        )
        h_gencut_met_pt = hist.Hist(
            dataset_axis, met_pt_axis, storage="weight", label="Counts"
        )
        h_gencut_met_fid_pt = hist.Hist(
            dataset_axis, met_pt_axis, storage="weight", label="Counts"
        )
        h_gencut_met_phi = hist.Hist(
            dataset_axis, met_phi_axis, storage="weight", label="Counts"
        )

        lhe_parts = events.LHEPart
        lhe_wgt = events.genWeight
        lhe_lep = lhe_parts[
            (
                (abs(lhe_parts.pdgId) == 11)
                | (abs(lhe_parts.pdgId) == 13)
                | (abs(lhe_parts.pdgId) == 15)
            )
            & (lhe_parts.status > -1)
        ]
        lhe_ele = lhe_parts[(abs(lhe_parts.pdgId) == 11) & (lhe_parts.status > -1)]
        lhe_mu = lhe_parts[(abs(lhe_parts.pdgId) == 13) & (lhe_parts.status > -1)]
        lhe_jet = lhe_parts[
            (
                (abs(lhe_parts.pdgId) > 0)
                | (abs(lhe_parts.pdgId) < 6)
                | (lhe_parts.pdgId == 21)
            )
            & (lhe_parts.status > -1)
        ]

        n_lhe_lep = ak.num(lhe_lep, axis=1)
        h_lhe_nlep.fill(dataset=dataset, x=n_lhe_lep, weight=np.sign(lhe_wgt))
        lhe_lep1 = lhe_lep[n_lhe_lep > 0][:, 0]
        lhe_lep1_wgt = lhe_wgt[n_lhe_lep > 0]
        h_lhe_l1_pt.fill(dataset=dataset, x=lhe_lep1.pt, weight=np.sign(lhe_lep1_wgt))
        h_lhe_l1_phi.fill(dataset=dataset, x=lhe_lep1.phi, weight=np.sign(lhe_lep1_wgt))
        h_lhe_l1_eta.fill(dataset=dataset, x=lhe_lep1.eta, weight=np.sign(lhe_lep1_wgt))

        n_lhe_ele = ak.num(lhe_ele, axis=1)
        lhe_ele1 = lhe_ele[n_lhe_ele > 0][:, 0]
        lhe_ele1_wgt = lhe_wgt[n_lhe_ele > 0]
        h_lhe_e1_pt.fill(dataset=dataset, x=lhe_ele1.pt, weight=np.sign(lhe_ele1_wgt))
        h_lhe_e1_phi.fill(dataset=dataset, x=lhe_ele1.phi, weight=np.sign(lhe_ele1_wgt))
        h_lhe_e1_eta.fill(dataset=dataset, x=lhe_ele1.eta, weight=np.sign(lhe_ele1_wgt))

        n_lhe_mu = ak.num(lhe_mu, axis=1)
        lhe_mu1 = lhe_mu[n_lhe_mu > 0][:, 0]
        lhe_mu1_wgt = lhe_wgt[n_lhe_mu > 0]
        h_lhe_mu1_pt.fill(dataset=dataset, x=lhe_mu1.pt, weight=np.sign(lhe_mu1_wgt))
        h_lhe_mu1_phi.fill(dataset=dataset, x=lhe_mu1.phi, weight=np.sign(lhe_mu1_wgt))
        h_lhe_mu1_eta.fill(dataset=dataset, x=lhe_mu1.eta, weight=np.sign(lhe_mu1_wgt))

        n_lhe_jet = ak.num(lhe_jet, axis=1)
        h_lhe_njet.fill(dataset=dataset, x=n_lhe_jet, weight=np.sign(lhe_wgt))
        lhe_jet1 = lhe_jet[n_lhe_jet > 0][:, 0]
        lhe_jet1_wgt = lhe_wgt[n_lhe_jet > 0]
        lhe_jet2 = lhe_jet[n_lhe_jet > 1][:, 1]
        lhe_jet2_wgt = lhe_wgt[n_lhe_jet > 1]
        lhe_jet3 = lhe_jet[n_lhe_jet > 2][:, 2]
        lhe_jet3_wgt = lhe_wgt[n_lhe_jet > 2]
        lhe_jet4 = lhe_jet[n_lhe_jet > 3][:, 3]
        lhe_jet4_wgt = lhe_wgt[n_lhe_jet > 3]
        h_lhe_j1_pt.fill(dataset=dataset, x=lhe_jet1.pt, weight=lhe_jet1_wgt)
        h_lhe_j2_pt.fill(dataset=dataset, x=lhe_jet2.pt, weight=lhe_jet2_wgt)
        h_lhe_j3_pt.fill(dataset=dataset, x=lhe_jet3.pt, weight=lhe_jet3_wgt)
        h_lhe_j4_pt.fill(dataset=dataset, x=lhe_jet4.pt, weight=lhe_jet4_wgt)
        h_lhe_j1_eta.fill(dataset=dataset, x=lhe_jet1.eta, weight=lhe_jet1_wgt)
        h_lhe_j2_eta.fill(dataset=dataset, x=lhe_jet2.eta, weight=lhe_jet2_wgt)
        h_lhe_j3_eta.fill(dataset=dataset, x=lhe_jet3.eta, weight=lhe_jet3_wgt)
        h_lhe_j4_eta.fill(dataset=dataset, x=lhe_jet4.eta, weight=lhe_jet4_wgt)
        h_lhe_j1_phi.fill(dataset=dataset, x=lhe_jet1.phi, weight=lhe_jet1_wgt)
        h_lhe_j2_phi.fill(dataset=dataset, x=lhe_jet2.phi, weight=lhe_jet2_wgt)
        h_lhe_j3_phi.fill(dataset=dataset, x=lhe_jet3.phi, weight=lhe_jet3_wgt)
        h_lhe_j4_phi.fill(dataset=dataset, x=lhe_jet4.phi, weight=lhe_jet4_wgt)
        h_lhe_j1_mass.fill(dataset=dataset, x=lhe_jet1.mass, weight=lhe_jet1_wgt)
        h_lhe_j2_mass.fill(dataset=dataset, x=lhe_jet2.mass, weight=lhe_jet2_wgt)
        h_lhe_j3_mass.fill(dataset=dataset, x=lhe_jet3.mass, weight=lhe_jet3_wgt)
        h_lhe_j4_mass.fill(dataset=dataset, x=lhe_jet4.mass, weight=lhe_jet4_wgt)

        gen_leps = events.GenDressedLepton
        gen_lep = gen_leps[(abs(gen_leps.pdgId) == 11) | (abs(gen_leps.pdgId) == 13)]
        gen_ele = gen_leps[(abs(gen_leps.pdgId) == 11)]
        gen_mu = gen_leps[(abs(gen_leps.pdgId) == 13)]

        n_gen_lep = ak.num(gen_lep, axis=1)
        h_gen_nlep.fill(dataset=dataset, x=n_gen_lep, weight=lhe_wgt)
        gen_lep1 = gen_lep[n_gen_lep > 0][:, 0]
        gen_lep1_wgt = lhe_wgt[n_gen_lep > 0]
        h_gen_l1_pt.fill(dataset=dataset, x=gen_lep1.pt, weight=gen_lep1_wgt)
        h_gen_l1_phi.fill(dataset=dataset, x=gen_lep1.phi, weight=gen_lep1_wgt)
        h_gen_l1_eta.fill(dataset=dataset, x=gen_lep1.eta, weight=gen_lep1_wgt)

        n_gen_ele = ak.num(gen_ele, axis=1)
        gen_ele1 = gen_ele[n_gen_ele > 0][:, 0]
        gen_ele1_wgt = lhe_wgt[n_gen_ele > 0]
        h_gen_e1_pt.fill(dataset=dataset, x=gen_ele1.pt, weight=gen_ele1_wgt)
        h_gen_e1_phi.fill(dataset=dataset, x=gen_ele1.phi, weight=gen_ele1_wgt)
        h_gen_e1_eta.fill(dataset=dataset, x=gen_ele1.eta, weight=gen_ele1_wgt)

        n_gen_mu = ak.num(gen_mu, axis=1)
        gen_mu1 = gen_mu[n_gen_mu > 0][:, 0]
        gen_mu1_wgt = lhe_wgt[n_gen_mu > 0]
        h_gen_mu1_pt.fill(dataset=dataset, x=gen_mu1.pt, weight=gen_mu1_wgt)
        h_gen_mu1_phi.fill(dataset=dataset, x=gen_mu1.phi, weight=gen_mu1_wgt)
        h_gen_mu1_eta.fill(dataset=dataset, x=gen_mu1.eta, weight=gen_mu1_wgt)

        gen_jets = events.GenJet
        n_gen_jet = ak.num(gen_jets, axis=1)
        h_gen_njet.fill(dataset=dataset, x=n_gen_jet, weight=lhe_wgt)
        gen_jet1 = gen_jets[n_gen_jet > 0][:, 0]
        gen_jet1_wgt = lhe_wgt[n_gen_jet > 0]
        gen_jet2 = gen_jets[n_gen_jet > 1][:, 1]
        gen_jet2_wgt = lhe_wgt[n_gen_jet > 1]
        gen_jet3 = gen_jets[n_gen_jet > 2][:, 2]
        gen_jet3_wgt = lhe_wgt[n_gen_jet > 2]
        gen_jet4 = gen_jets[n_gen_jet > 3][:, 3]
        gen_jet4_wgt = lhe_wgt[n_gen_jet > 3]
        h_gen_j1_pt.fill(dataset=dataset, x=gen_jet1.pt, weight=gen_jet1_wgt)
        h_gen_j2_pt.fill(dataset=dataset, x=gen_jet2.pt, weight=gen_jet2_wgt)
        h_gen_j3_pt.fill(dataset=dataset, x=gen_jet3.pt, weight=gen_jet3_wgt)
        h_gen_j4_pt.fill(dataset=dataset, x=gen_jet4.pt, weight=gen_jet4_wgt)
        h_gen_j1_eta.fill(dataset=dataset, x=gen_jet1.eta, weight=gen_jet1_wgt)
        h_gen_j2_eta.fill(dataset=dataset, x=gen_jet2.eta, weight=gen_jet2_wgt)
        h_gen_j3_eta.fill(dataset=dataset, x=gen_jet3.eta, weight=gen_jet3_wgt)
        h_gen_j4_eta.fill(dataset=dataset, x=gen_jet4.eta, weight=gen_jet4_wgt)
        h_gen_j1_phi.fill(dataset=dataset, x=gen_jet1.phi, weight=gen_jet1_wgt)
        h_gen_j2_phi.fill(dataset=dataset, x=gen_jet2.phi, weight=gen_jet2_wgt)
        h_gen_j3_phi.fill(dataset=dataset, x=gen_jet3.phi, weight=gen_jet3_wgt)
        h_gen_j4_phi.fill(dataset=dataset, x=gen_jet4.phi, weight=gen_jet4_wgt)
        h_gen_j1_mass.fill(dataset=dataset, x=gen_jet1.mass, weight=gen_jet1_wgt)
        h_gen_j2_mass.fill(dataset=dataset, x=gen_jet2.mass, weight=gen_jet2_wgt)
        h_gen_j3_mass.fill(dataset=dataset, x=gen_jet3.mass, weight=gen_jet3_wgt)
        h_gen_j4_mass.fill(dataset=dataset, x=gen_jet4.mass, weight=gen_jet4_wgt)

        h_gen_met_pt.fill(dataset=dataset, x=events.GenMET.pt, weight=lhe_wgt)
        h_gen_met_fid_pt.fill(
            dataset=dataset, x=events.MET.fiducialGenPt, weight=lhe_wgt
        )
        h_gen_met_phi.fill(dataset=dataset, x=events.GenMET.phi, weight=lhe_wgt)

        # with gen selections
        gleps = events.GenDressedLepton
        leps = gleps[
            (abs(gleps.pdgId) == 11)
            | (abs(gleps.pdgId) == 13) & (gleps.pt > genlep_pt_min)
        ]
        eles = gleps[(abs(gleps.pdgId) == 11) & (gleps.pt > genlep_pt_min)]
        mus = gleps[(abs(gleps.pdgId) == 13) & (gleps.pt > genlep_pt_min)]

        n_lep = ak.num(leps, axis=1)
        h_gencut_nlep.fill(dataset=dataset, x=n_lep, weight=lhe_wgt)
        leps = leps[n_lep > 0]
        lep1 = leps[:, 0]
        lep1_wgt = lhe_wgt[n_lep > 0]
        met = events.GenMET[n_lep > 0]
        met_fid = events.MET[n_lep > 0]
        h_gencut_l1_pt.fill(dataset=dataset, x=lep1.pt, weight=lep1_wgt)
        h_gencut_l1_phi.fill(dataset=dataset, x=lep1.phi, weight=lep1_wgt)
        h_gencut_l1_eta.fill(dataset=dataset, x=lep1.eta, weight=lep1_wgt)
        h_gencut_met_pt.fill(dataset=dataset, x=met.pt, weight=lep1_wgt)
        h_gencut_met_fid_pt.fill(
            dataset=dataset, x=met_fid.fiducialGenPt, weight=lep1_wgt
        )
        h_gencut_met_phi.fill(dataset=dataset, x=met.phi, weight=lep1_wgt)

        n_ele = ak.num(eles, axis=1)
        eles = eles[n_ele > 0]
        ele1 = eles[:, 0]
        ele1_wgt = lhe_wgt[n_ele > 0]
        h_gencut_e1_pt.fill(dataset=dataset, x=ele1.pt, weight=ele1_wgt)
        h_gencut_e1_phi.fill(dataset=dataset, x=ele1.phi, weight=ele1_wgt)
        h_gencut_e1_eta.fill(dataset=dataset, x=ele1.eta, weight=ele1_wgt)

        n_mu = ak.num(mus, axis=1)
        mus = mus[n_mu > 0]
        mu1 = mus[:, 0]
        mu1_wgt = lhe_wgt[n_mu > 0]
        h_gencut_mu1_pt.fill(dataset=dataset, x=mu1.pt, weight=mu1_wgt)
        h_gencut_mu1_phi.fill(dataset=dataset, x=mu1.phi, weight=mu1_wgt)
        h_gencut_mu1_eta.fill(dataset=dataset, x=mu1.eta, weight=mu1_wgt)

        gjets = events.GenJet[n_lep > 0]
        jet_lep_matcher, jet_lep_dr = gjets.nearest(leps, axis=1, return_metric=True)
        jets = gjets[(gjets.pt > genjet_pt_min) & (jet_lep_dr > lep_jet_dr)]
        n_jet = ak.num(jets, axis=1)
        n_jet = ak.fill_none(n_jet, 0)
        jet_wgt = lhe_wgt[n_lep > 0]
        h_gencut_njet.fill(dataset=dataset, x=n_jet, weight=jet_wgt)

        jet1 = jets[n_jet > 0][:, 0]
        jet1_wgt = jet_wgt[n_jet > 0]
        jet2 = jets[n_jet > 1][:, 1]
        jet2_wgt = jet_wgt[n_jet > 1]
        jet3 = jets[n_jet > 2][:, 2]
        jet3_wgt = jet_wgt[n_jet > 2]
        jet4 = jets[n_jet > 3][:, 3]
        jet4_wgt = jet_wgt[n_jet > 3]
        h_gencut_j1_pt.fill(dataset=dataset, x=jet1.pt, weight=jet1_wgt)
        h_gencut_j2_pt.fill(dataset=dataset, x=jet2.pt, weight=jet2_wgt)
        h_gencut_j3_pt.fill(dataset=dataset, x=jet3.pt, weight=jet3_wgt)
        h_gencut_j4_pt.fill(dataset=dataset, x=jet4.pt, weight=jet4_wgt)
        h_gencut_j1_eta.fill(dataset=dataset, x=jet1.eta, weight=jet1_wgt)
        h_gencut_j2_eta.fill(dataset=dataset, x=jet2.eta, weight=jet2_wgt)
        h_gencut_j3_eta.fill(dataset=dataset, x=jet3.eta, weight=jet3_wgt)
        h_gencut_j4_eta.fill(dataset=dataset, x=jet4.eta, weight=jet4_wgt)
        h_gencut_j1_phi.fill(dataset=dataset, x=jet1.phi, weight=jet1_wgt)
        h_gencut_j2_phi.fill(dataset=dataset, x=jet2.phi, weight=jet2_wgt)
        h_gencut_j3_phi.fill(dataset=dataset, x=jet3.phi, weight=jet3_wgt)
        h_gencut_j4_phi.fill(dataset=dataset, x=jet4.phi, weight=jet4_wgt)
        h_gencut_j1_mass.fill(dataset=dataset, x=jet1.mass, weight=jet1_wgt)
        h_gencut_j2_mass.fill(dataset=dataset, x=jet2.mass, weight=jet2_wgt)
        h_gencut_j3_mass.fill(dataset=dataset, x=jet3.mass, weight=jet3_wgt)
        h_gencut_j4_mass.fill(dataset=dataset, x=jet4.mass, weight=jet4_wgt)

        return {
            "h_lhe_nlep": h_lhe_nlep,
            "h_lhe_e1_pt": h_lhe_e1_pt,
            "h_lhe_e1_eta": h_lhe_e1_eta,
            "h_lhe_e1_phi": h_lhe_e1_phi,
            "h_lhe_mu1_pt": h_lhe_mu1_pt,
            "h_lhe_mu1_eta": h_lhe_mu1_eta,
            "h_lhe_mu1_phi": h_lhe_mu1_phi,
            "h_lhe_l1_pt": h_lhe_l1_pt,
            "h_lhe_l1_eta": h_lhe_l1_eta,
            "h_lhe_l1_phi": h_lhe_l1_phi,
            "h_lhe_njet": h_lhe_njet,
            "h_lhe_j1_pt": h_lhe_j1_pt,
            "h_lhe_j2_pt": h_lhe_j2_pt,
            "h_lhe_j3_pt": h_lhe_j3_pt,
            "h_lhe_j4_pt": h_lhe_j4_pt,
            "h_lhe_j1_eta": h_lhe_j1_eta,
            "h_lhe_j2_eta": h_lhe_j2_eta,
            "h_lhe_j3_eta": h_lhe_j3_eta,
            "h_lhe_j4_eta": h_lhe_j4_eta,
            "h_lhe_j1_phi": h_lhe_j1_phi,
            "h_lhe_j2_phi": h_lhe_j2_phi,
            "h_lhe_j3_phi": h_lhe_j3_phi,
            "h_lhe_j4_phi": h_lhe_j4_phi,
            "h_lhe_j1_mass": h_lhe_j1_mass,
            "h_lhe_j2_mass": h_lhe_j2_mass,
            "h_lhe_j3_mass": h_lhe_j3_mass,
            "h_lhe_j4_mass": h_lhe_j4_mass,
            "h_gen_nlep": h_gen_nlep,
            "h_gen_e1_pt": h_gen_e1_pt,
            "h_gen_e1_eta": h_gen_e1_eta,
            "h_gen_e1_phi": h_gen_e1_phi,
            "h_gen_mu1_pt": h_gen_mu1_pt,
            "h_gen_mu1_eta": h_gen_mu1_eta,
            "h_gen_mu1_phi": h_gen_mu1_phi,
            "h_gen_l1_pt": h_gen_l1_pt,
            "h_gen_l1_eta": h_gen_l1_eta,
            "h_gen_l1_phi": h_gen_l1_phi,
            "h_gen_njet": h_gen_njet,
            "h_gen_j1_pt": h_gen_j1_pt,
            "h_gen_j2_pt": h_gen_j2_pt,
            "h_gen_j3_pt": h_gen_j3_pt,
            "h_gen_j4_pt": h_gen_j4_pt,
            "h_gen_j1_eta": h_gen_j1_eta,
            "h_gen_j2_eta": h_gen_j2_eta,
            "h_gen_j3_eta": h_gen_j3_eta,
            "h_gen_j4_eta": h_gen_j4_eta,
            "h_gen_j1_phi": h_gen_j1_phi,
            "h_gen_j2_phi": h_gen_j2_phi,
            "h_gen_j3_phi": h_gen_j3_phi,
            "h_gen_j4_phi": h_gen_j4_phi,
            "h_gen_j1_mass": h_gen_j1_mass,
            "h_gen_j2_mass": h_gen_j2_mass,
            "h_gen_j3_mass": h_gen_j3_mass,
            "h_gen_j4_mass": h_gen_j4_mass,
            "h_gen_met_pt": h_gen_met_pt,
            "h_gen_met_fid_pt": h_gen_met_fid_pt,
            "h_gen_met_phi": h_gen_met_phi,
            "h_gencut_nlep": h_gencut_nlep,
            "h_gencut_e1_pt": h_gencut_e1_pt,
            "h_gencut_e1_eta": h_gencut_e1_eta,
            "h_gencut_e1_phi": h_gencut_e1_phi,
            "h_gencut_mu1_pt": h_gencut_mu1_pt,
            "h_gencut_mu1_eta": h_gencut_mu1_eta,
            "h_gencut_mu1_phi": h_gencut_mu1_phi,
            "h_gencut_l1_pt": h_gencut_l1_pt,
            "h_gencut_l1_eta": h_gencut_l1_eta,
            "h_gencut_l1_phi": h_gencut_l1_phi,
            "h_gencut_njet": h_gencut_njet,
            "h_gencut_j1_pt": h_gencut_j1_pt,
            "h_gencut_j2_pt": h_gencut_j2_pt,
            "h_gencut_j3_pt": h_gencut_j3_pt,
            "h_gencut_j4_pt": h_gencut_j4_pt,
            "h_gencut_j1_eta": h_gencut_j1_eta,
            "h_gencut_j2_eta": h_gencut_j2_eta,
            "h_gencut_j3_eta": h_gencut_j3_eta,
            "h_gencut_j4_eta": h_gencut_j4_eta,
            "h_gencut_j1_phi": h_gencut_j1_phi,
            "h_gencut_j2_phi": h_gencut_j2_phi,
            "h_gencut_j3_phi": h_gencut_j3_phi,
            "h_gencut_j4_phi": h_gencut_j4_phi,
            "h_gencut_j1_mass": h_gencut_j1_mass,
            "h_gencut_j2_mass": h_gencut_j2_mass,
            "h_gencut_j3_mass": h_gencut_j3_mass,
            "h_gencut_j4_mass": h_gencut_j4_mass,
            "h_gencut_met_pt": h_gencut_met_pt,
            "h_gencut_met_fid_pt": h_gencut_met_fid_pt,
            "h_gencut_met_phi": h_gencut_met_phi,
            "count": {dataset: count_dict},
        }

    def postprocess(self, accumulator):
        pass


if __name__ == "__main__":
    path_off = "/eos/lyoeos.in2p3.fr/grid/cms/store/mc/Run3Summer22EEwmLHENanoGEN/WtoLNu-2Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_readonlyTest/NANOAODSIM/124X_mcRun3_2022_realistic_postEE_v1-v1/80000"
    path_con = "/gridgroup/cms/jxiao/genval/wjet/out"
    flist = get_file_list(path_off, "*.root")
    flist1 = []
    flist2 = []
    for i, fname in enumerate(flist):
        if i % 2 == 0:
            flist1.append(fname)
        else:
            flist2.append(fname)
    # fileset = {
    #     "official": get_file_list(path_off, "*.root"),
    #     "condor": get_file_list(path_off, "*.root"),
    # }
    fileset = {"official": flist1, "condor": flist2}
    # fsig = "/eos/lyoeos.in2p3.fr/grid/cms/store/mc/Run3Summer22EEwmLHENanoGEN/WtoLNu-2Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_readonlyTest/NANOAODSIM/124X_mcRun3_2022_realistic_postEE_v1-v1/2810000/6965c17d-55f2-4c85-8b3e-9c6d8be8ff99.root"
    # fbkg = "/eos/lyoeos.in2p3.fr/grid/cms/store/mc/Run3Summer22EEwmLHENanoGEN/WtoLNu-2Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8_readonlyTest/NANOAODSIM/124X_mcRun3_2022_realistic_postEE_v1-v1/2810000/f0ef893d-e097-474a-851a-dcbf400fabc0.root"
    # fileset = {"official": [fsig], "condor": [fbkg]}

    executor = processor.FuturesExecutor(workers=72)

    run = processor.Runner(
        executor=executor,
        schema=processor.NanoAODSchema,
        chunksize=10000,
        maxchunks=3000,
        # format=_args.format,
        # skipbadfiles=_args.skipbadfiles,
    )
    toc = time.monotonic()
    output = run(
        fileset,
        treename="Events",
        processor_instance=WJetProcessor(),
    )
    tic = time.monotonic()
    print(
        """
=======================
 Official files: {}
 Condor files: {}
 Total time: {} s
=======================
        """.format(
            len(fileset["official"]), len(fileset["condor"]), np.round(tic - toc, 4)
        )
    )

    outname = "rlts_seed_comparison.coffea"
    coffea.util.save(output, outname)
    print("[WJets] Save results in {}".format(outname))
