#include "OpNoviceEventAction.hh"
#include "stdio.h"
#include "DataFileManager.hh"
#include "dataOpt.hh"
#include <G4SystemOfUnits.hh>

OpNoviceEventAction::OpNoviceEventAction(int x, int y)
        : G4UserEventAction()
{
    std::string filename_start = "GammaCamera";
    std::string filename = filename_start + "_x_" + std::to_string((int)x) + "_y_" + std::to_string((int)y) + "_copynum_";
    foutGammaCamera = DataFileManager::instance()->getDataFile<SensitiveDetectorParticleDataOpt>(filename);
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