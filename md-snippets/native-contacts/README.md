# Native contacts over a Molecular Dynamics trajectory

`globalq.py` calculates the fraction of native contacts (Q) based on the tutorial provided in the MDTraj documentation, based on the definition given by Best, Hummer, & Eaton (2013), which states:

for an all-atoms model in a given conformation (X), heavy atoms i and j belonging to residues θi and θj are in
contact provided that they are four residues apart (|θi – θj| > 3) with a cut-off distance of 4.5 Å
between atoms i and j

`perresq.py` performs the exact partition of Q into a per-residue basis, using the same criteria. 

It defines the native contacts set (S) on the first frame of the trajectory and each contact is assigned the same switching function as before
For each frame, the contact list was traversed once and each fij accumulated into both participating residues. 
For each residue a, qa is the mean of this function over the subset of native contacts in which the residue participates, 
normalized by its own native contact count. Intra-residue contacts were naturally excluded by the  | θi-θj | > 3 criterion, 
so global Q can be recovered as a contact-count-weighted mean of qa as means of validation.

**Usage**
    python perresq.py -f md.xtc -s md.gro -o output_name
