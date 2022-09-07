//
// Created by zelenyy on 10.01.19.
//

#ifndef GEANT4_THUNDERSTORM_OPTSTEPPINGACTION_HH
#define GEANT4_THUNDERSTORM_OPTSTEPPINGACTION_HH


#include <G4UserSteppingAction.hh>
#include <dataOpt.hh>
#include <DataFile.hh>
//#include <random>





class OpNoviceSteppingAction: public G4UserSteppingAction {
public:
    OpNoviceSteppingAction();
    virtual ~OpNoviceSteppingAction();
    void UserSteppingAction(const G4Step *step) override;
    G4double results[4];
private:
    


};




#endif //GEANT4_THUNDERSTORM_OPTSTEPPINGACTION_HH