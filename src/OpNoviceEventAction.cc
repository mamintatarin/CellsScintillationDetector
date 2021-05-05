#include "OpNoviceEventAction.hh"
#include "stdio.h"
#include "DataFileManager.hh"
#include "dataOpt.hh"
OpNoviceEventAction::OpNoviceEventAction()
        : G4UserEventAction()
{
    foutGammaCamera = DataFileManager::instance()->getDataFile<SensitiveDetectorParticleDataOpt>("GammaCamera");
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

OpNoviceEventAction::~OpNoviceEventAction()
{

}
void OpNoviceEventAction::EndOfEventAction(const G4Event* Event)
{

    G4int event = G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID();

    FillParticleDataOpt(data,position.getX(),position.getY(),
                        tempStepping->results[0],tempStepping->results[1]
            ,tempStepping->results[2],tempStepping->results[3]);

    foutGammaCamera->addData(data);


}

void OpNoviceEventAction::BeginOfEventAction(const G4Event* Event)
{
    tempStepping->results[0]=0.0;
    tempStepping->results[1]=0.0;
    tempStepping->results[2]=0.0;
    tempStepping->results[3]=0.0;
}