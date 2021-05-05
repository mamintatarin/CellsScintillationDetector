#ifndef OpNoviceStackingAction_H
#define OpNoviceStackingAction_H 1

#include "globals.hh"
#include "G4UserStackingAction.hh"
#include "DataFileManager.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

class OpNoviceStackingAction : public G4UserStackingAction
{
  public:
    OpNoviceStackingAction();
    virtual ~OpNoviceStackingAction();

  public:
    virtual G4ClassificationOfNewTrack ClassifyNewTrack(const G4Track* aTrack);
    virtual void NewStage();
    virtual void PrepareNewEvent();

};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif