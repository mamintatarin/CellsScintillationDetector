#ifndef OpNoviceRunAction_h
#define OpNoviceRunAction_h 1

#include <G4INCLThreeVector.hh>
#include "globals.hh"
#include "G4UserRunAction.hh"
#include <CLHEP/Vector/ThreeVector.h>

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......


class G4Run;

class OpNoviceRunAction : public G4UserRunAction
{
  public:
    OpNoviceRunAction(CLHEP::Hep3Vector pos);
    virtual ~OpNoviceRunAction();

  public:
    virtual void BeginOfRunAction(const G4Run* aRun);
    virtual void EndOfRunAction(const G4Run* aRun);

 
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif /*OpNoviceRunAction_h*/