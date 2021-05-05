#ifndef OpNoviceDetectorConstruction_h
#define OpNoviceDetectorConstruction_h 1

#include "globals.hh"
#include "G4VUserDetectorConstruction.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

class OpNoviceDetectorConstruction : public G4VUserDetectorConstruction
{
  public:
    OpNoviceDetectorConstruction();
    virtual ~OpNoviceDetectorConstruction();

  public:
    G4double side;
    G4double roof;
    virtual G4VPhysicalVolume* Construct();

 
};

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

#endif /*OpNoviceDetectorConstruction_h*/