PROMPTS_ASR_RAW = """Begin by converting the spoken words into written text. 
Can you transcribe the speech into a written format? 
Focus on translating the audible content into text. 
Transcribe the speech by carefully listening to it.
Would you kindly write down the content of the speech? 
Analyze the speech and create a written transcription.
Transcribe the speech by writing down the spoken words. 
Can you write down what is being said in the speech? 
Pay attention to the speech and transcribe it into text. 
Write down the content of the speech as you listen to it. 
Can you create a written transcription of the speech? 
Listen carefully to the speech and transcribe it into text. 
Transcribe the speech by writing down what is being said.
Can you write down the spoken words in the speech? 
Focus on the content of the speech and transcribe it into text. 
Write down what is being said in the speech as you listen to it.
Can you create a written record of the speech? 
Listen to the speech and transcribe it into text by writing down what is being said.
Transcribe the speech by carefully listening and writing down the spoken words. 
Can you write down the content of the speech as you listen to it? 
Pay attention to what is being said in the speech and transcribe it into text.
Write down the spoken words in the speech as you listen to it. 
Can you create a written transcription of what is being said in the speech? 
Listen carefully to the content of the speech and transcribe it into text. 
Transcribe the speech by writing down what you hear as you listen to it.
Listen to the speech and write down the spoken words as you hear them. 
Transcribe the speech by paying attention to what is being said and writing it down. 
Can you create a written record of the content of the speech? 
Focus on the spoken words in the speech and transcribe them into text. 
Write down what you hear as you listen to the speech. 
Can you transcribe the speech by writing down what is being said? 
Listen carefully to the speech and write down the spoken words. 
Transcribe the speech by paying attention to the content and writing it down. 
Can you write down what you hear in the speech? 
Focus on what is being said in the speech and transcribe it into text. 
Write down the content of the speech as you listen carefully to it.
Can you create a written transcription of the spoken words in the speech? 
Transcribe the speech by listening carefully and writing down what you hear. 
Can you write down the content of the speech as it is being spoken? 
Pay attention to the spoken words in the speech and transcribe them into text.
Convert the spoken words into written text.
Transcribe the speech by carefully listening to it.
Analyze the speech and create a written transcription.
Write down the content of the speech.
Translate the audible content into text.
Record the spoken words and convert them into text.
Use speech recognition to transcribe the audio.
Listen to the speech and write down what you hear.
Create a written transcript of the speech.
Use automatic speech recognition to transcribe the audio.
Write down what is being said in the speech.
Transcribe the spoken words into text format.
Convert the audio into written text using speech recognition.
Listen to the audio and write down what you hear.
Use voice recognition to transcribe the speech into text format.
Create a written record of what is being said in the audio.
Use machine learning to transcribe the spoken words into text format.
Write down what you hear in the audio recording.
Use natural language processing to transcribe the speech into text format.
Create a written transcript of what is being said in the audio recording.
Write what you hear from the speech.
Identify the words that are spoken in the speech and write them down.
Listen to the speech and transcribe it into text.
What is the content of the speech? Write it down.
Convert the speech into written form by listening carefully.
Transcribe the audible words into text format.
Write down the speech as you hear it.
How would you write the speech in text? Transcribe it.
Listen and write what the speaker is saying.
Transcribe the speech by writing down the spoken words.
Write what you hear from the speech in text form.
How would you express the speech in written words?
Listen to the speech and type out the content.
What is the speech saying? Write it down.
Convert the speech into text by listening carefully.
Type what the speech is saying in text form.
How would you transcribe the speech into writing?
Write down the content of the speech as text.
Listen to the speech and write what you hear.
What is the content of the speech? Transcribe it in text form.
Convert the spoken words into written text.
Transcribe the speech by carefully listening to it.
Analyze the speech and create a written transcription.
Write down the content of the speech.
Translate the audible content into text.
Can you transcribe the speech into a written format?
Would you kindly write down the content of the speech?
Focus on translating the audible content into text.
Can you write down what was said?
Transcribe the speech without any errors.
Write down what you hear.
Can you convert the spoken words into text?
Create a written transcription of the speech.
Can you write down what was spoken?
Transcribe the speech accurately.
Write down what you hear in the speech.
Can you transcribe what was said?
Convert the audible content into text.
Write down what was said in the speech.
Can you create a written transcription of what was spoken?"""

PROMPTS_ASR = []
for i in PROMPTS_ASR_RAW.split("\n"):
    PROMPTS_ASR.append(i)
debug = 1