# More than one possible keyword can be used in an incident description, so the primary ones are to be checked first, and if there are no matches, then the secondary ones
# This is to avoid some cases being misclassified - ex. a sexual assault case labeled as "Assault" because it has the string "assault" in the description.
primary_keywords = {
    'sexual assault' : "Sexual Assault",
    'deceased' : "Homicide",
}
secondary_keywords = {
    'assault' : "Assault",
    'aggravated' : "Assault",
    'fentanyl' : "Drug Abuse",
    'overdose' : "Drug Abuse",
    'overdoses' : "Drug Abuse",
    'child abduction' : "Child Abduction",
    'shooting' : "Firearms",
    'firearm' : "Firearms",
    'firearms' : "Firearms",
    'stabbing' : "Stabbing",
    'human trafficking' : "Human Trafficking",
    'robbery' : "Robbery",
    'online' : "Online Activity"
}