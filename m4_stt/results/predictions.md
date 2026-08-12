# Whisper STT (Module 4) Final Predictions

Here are the actual comparisons generated from the evaluation script. 
By saving them as text, your operating system will natively render the Telugu font correctly.

### Sample 1
**ACTUAL:**    సుదీర్ఘకాలం పనిచేసిన తర్వాత వారు ఫ్రాంక్లిన్‌లోని తమ ఇంటిలో పదవీ విరమణ చేశారు
**PREDICTED:** స్ర్ర్ ద్ర్స్త్ర్ ప్రాత్నినిలో ముర్న్నినాలో చేరాయా స్రాయాయాయాయాయాయాయాయా ప్రియాయాయాయాయాయాయా

### Sample 2
**ACTUAL:**    వేరే దేశంలో జీవించడం మరియు స్వచ్చంధంగా ఉండటం అనేది విభిన్న సంస్కృతిని తెలుసుకోవడానికి కొత్త వ్యక్తులను కలుసుకోవడానికి మీ గురించి తెలుసుకోవడానికి మీ దృక్కోణాన్ని పొందడానికి మరియు కొత్త నైపుణ్యాలను పొందడానికి ఒక గొప్ప మార్గం
**PREDICTED:** వర్కి దాస్లో లోవిక్నిని సాచనిదాని అనిని అనిని వర్రిని సంవిని దిరిని దిరిని దిరిరిని రిరిరిని రిరిరిని రిరిరిని రిరిరిని రిరిరిని రిరిరిని

> **Note:** The repetition/looping of characters at the end of the prediction is a known characteristic of Whisper models when fine-tuned for a very small number of steps (like the 50 steps we used for quick CPU testing).
