#ifndef OpNoviceEventAction_h
#define OpNoviceEventAction_h 1

#include "G4Event.hh"
#include <G4UserEventAction.hh>
#include "OpNoviceSteppingAction.hh"
#include "OpNoviceRunAction.hh"
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......


class G4Event;

class OpNoviceEventAction: public G4UserEventAction
{
  public:
    OpNoviceEventAction(int x,int y);
    virtual ~OpNoviceEventAction();
    OpNoviceSteppingAction * tempStepping;

    CLHEP::Hep3Vector position;
  public:
    virtual void BeginOfEventAction(const G4Event* Event);
    virtual void EndOfEventAction(const G4Event* Event);
    SensitiveDetectorParticleDataOpt data;
    DataFile<SensitiveDetectorParticleDataOpt>* foutGammaCamera;

};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif /*OpNoviceEventAction_h*/