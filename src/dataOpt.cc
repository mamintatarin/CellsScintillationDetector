#include <dataOpt.hh>
#include <stdio.h>
#include "G4SystemOfUnits.hh"


using namespace CLHEP;





void FillParticleDataOpt(SensitiveDetectorParticleDataOpt &data, G4double x,G4double y,
                         G4double energy1,G4double energy2,G4double energy3,G4double energy4) {



    data.x=x;
    data.y=y;
    data.event=G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID();
    data.energy1 = energy1;
    data.energy2 = energy2;
    data.energy3 = energy3;
    data.energy4 = energy4;

}









