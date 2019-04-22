# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step3 --datatier GEN-SIM-RECO,MINIAODSIM,DQMIO --runUnscheduled --conditions auto:run1_mc -s RAW2DIGI,L1Reco,RECO,RECOSIM,EI,PAT,VALIDATION:@standardValidationNoHLT+@miniAODValidation,DQM:@standardDQMFakeHLT+@miniAODDQM --eventcontent RECOSIM,MINIAODSIM,DQM -n 100 --filein file:step2.root --fileout file:step3.root
import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras
import FWCore.ParameterSet.VarParsing as VarParsing

from glob import glob

options = VarParsing.VarParsing ('analysis')
process = cms.Process('ana',eras.Run2_2018)

# import of standard configurations
# process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
# process.load('Configuration.EventContent.EventContent_cff')
# process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
# process.load('Configuration.StandardSequences.MagneticField_cff')
# process.load('Configuration.StandardSequences.RawToDigi_cff')
# process.load('Configuration.StandardSequences.L1Reco_cff')
# process.load('Configuration.StandardSequences.Reconstruction_cff')
# process.load('Configuration.StandardSequences.RecoSim_cff')
# process.load('CommonTools.ParticleFlow.EITopPAG_cff')
# process.load('PhysicsTools.PatAlgos.slimming.metFilterPaths_cff')
# process.load('Configuration.StandardSequences.PATMC_cff')
# process.load('Configuration.StandardSequences.Validation_cff')
# process.load('DQMOffline.Configuration.DQMOfflineMC_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

options.maxEvents = 100000 ## -1 means all events

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(options.maxEvents)
)

# Input source
Chrg_Dir_Num = 4 # 0,1,2,3,4
Chrg_Pi_path = '/eos/uscms/store/user/hcal_upgrade/hatake/step3/PGun_step3_RECO_10_4_0_E2_500_v5/SinglePi/PGun_step3_RECO_10_4_0_E2_500_v5/190120_223058/000'+str(Chrg_Dir_Num)+'/step*.root'
options.inputFiles = ['file:'+name for name in glob(Chrg_Pi_path) if 'inMINIAODSIM' not in name and 'step3_12.root' not in name] ###### lpc only
#options.outputFile = '/eos/uscms/store/user/bcaraway/SinglePi/singlePi_trees_'+str(Chrg_Dir_Num)+'.root'
options.outputFile = 'test.root' 
#options.outputFile = 'test.root'
#print  (options.inputFiles if 'step3_12.root' in options.inputFiles else 'free of bug files!')
process.source = cms.Source("PoolSource",
    #fileNames = cms.untracked.vstring('root://se01.indiacms.res.in//store/user/spandey/step2/PGun_step2_DIGI_1002_2_200_Feb_12/CRAB_UserFiles/crab_PGun_step2_DIGI_1002_2_200_Feb_12/180212_110432/0000/step2_2.root'),
    fileNames = cms.untracked.vstring(options.inputFiles),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(
)

####################

from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run1_mc', '')
#process.GlobalTag = GlobalTag(process.GlobalTag, '100X_upgrade2018_realistic_v10', '')
#process.GlobalTag = GlobalTag(process.GlobalTag, '100X_mc2017_realistic_v3', '')
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2018_realistic', '')

process.pfChargedHadronAnalyzer = cms.EDAnalyzer(
    "PFChargedHadronAnalyzer",
    PFCandidates = cms.InputTag("particleFlow"),
    PFSimParticles = cms.InputTag("particleFlowSimParticle"),
    EcalPFClusters = cms.InputTag("particleFlowClusterECAL"),
    HcalPFClusters = cms.InputTag("particleFlowClusterHCAL"),
    EcalRecHitsEB    = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
    EcalRecHitsEE    = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
    EcalRecHitsES    = cms.InputTag("ecalPreshowerRecHit","EcalRecHitsES"),
    HcalRecHits    = cms.InputTag("hbhereco",""),
    #KH
    useCalibrationsFromDB = cms.bool(True),
    calibrationsLabel = cms.string(''),
    #KH
    ptMin = cms.double(1.),                     # Minimum pt
    pMin = cms.double(1.),                      # Minimum p
    nPixMin = cms.int32(2),                     # Nb of pixel hits
    nHitMin = cms.vint32(14,17,20,17,10),       # Nb of track hits
    nEtaMin = cms.vdouble(1.4,1.6,2.0,2.4,2.6,2.8,2.9,3.0), # in these eta ranges
    hcalMin = cms.double(0.5),                  # Minimum hcal energy
    ecalMax = cms.double(1E9),                  # Maximum ecal energy
    verbose = cms.untracked.bool(True),         # not used.
    #rootOutputFile = cms.string("PGun__2_200GeV__81X_upgrade2017_realistic_v22.root"),# the root tree
    rootOutputFile = cms.string(options.outputFile),# the root tree
    #IsMinBias = cms.untracked.bool(False)
)

process.load("RecoParticleFlow.PFProducer.particleFlowSimParticle_cfi")
#process.load("RecoParticleFlow.Configuration.HepMCCopy_cfi")

from FastSimulation.Event.ParticleFilter_cfi import  ParticleFilterBlock
process.particleFlowSimParticle.ParticleFilter = ParticleFilterBlock.ParticleFilter.copy()
process.particleFlowSimParticle.ParticleFilter.chargedPtMin = cms.double(0.0)
process.particleFlowSimParticle.ParticleFilter.EMin = cms.double(0.0)
#process.particleFlowSimParticle.ParticleFilter = cms.PSet(
#        # Allow *ALL* protons with energy > protonEMin
#        protonEMin = cms.double(5000.0),
#        # Particles must have abs(eta) < etaMax (if close enough to 0,0,0)
#        etaMax = cms.double(5.3),
#        # Charged particles with pT < pTMin (GeV/c) are not simulated
#        chargedPtMin = cms.double(0.0),
#        # Particles must have energy greater than EMin [GeV]
#        EMin = cms.double(0.0)
#)

process.genReReco = cms.Sequence(
    process.particleFlowSimParticle
)

# Path and EndPath definitions

process.EDA = cms.EndPath(process.pfChargedHadronAnalyzer)
process.gRR = cms.EndPath(process.genReReco)

process.schedule = cms.Schedule(process.gRR,process.EDA)
#process.schedule = cms.Schedule(process.EDA)

