import json
import requests


# Configure URL and Desired Endpoint
endpoint = '/summarize' #in base could pick from: ["/summarize", "embedding", "keyphrase"]
url = "YOUR IP ADDRESS GOES HERE" + endpoint


# Sample Text for Request
payload = json.dumps({
  "input_text": "Jazz is a music genre that originated in the African-American communities of New Orleans, Louisiana in the late 19th and early 20th centuries, with its roots in blues and ragtime.[1][2][3][4] Since the 1920s Jazz Age, it has been recognized as a major form of musical expression in traditional and popular music. Jazz is characterized by swing and blue notes, complex chords, call and response vocals, polyrhythms and improvisation. Jazz has roots in European harmony and African rhythmic rituals.[5][6]\n\nAs jazz spread around the world, it drew on national, regional, and local musical cultures, which gave rise to different styles. New Orleans jazz began in the early 1910s, combining earlier brass-band marches, French quadrilles, biguine, ragtime and blues with collective polyphonic improvisation. But jazz did not begin as a single musical tradition in New Orleans or elsewhere.[7] In the 1930s, arranged dance-oriented swing big bands, Kansas City jazz (a hard-swinging, bluesy, improvisational style), and gypsy jazz (a style that emphasized musette waltzes) were the prominent styles. Bebop emerged in the 1940s, shifting jazz from danceable popular music toward a more challenging \"musician's music\" which was played at faster tempos and used more chord-based improvisation. Cool jazz developed near the end of the 1940s, introducing calmer, smoother sounds and long, linear melodic lines.[8]\n\nThe mid-1950s saw the emergence of hard bop, which introduced influences from rhythm and blues, gospel, and blues to small groups and particularly to saxophone and piano. Modal jazz developed in the late 1950s, using the mode, or musical scale, as the basis of musical structure and improvisation, as did free jazz, which explored playing without regular meter, beat and formal structures. Jazz-rock fusion appeared in the late 1960s and early 1970s, combining jazz improvisation with rock music's rhythms, electric instruments, and highly amplified stage sound. In the early 1980s, a commercial form of jazz fusion called smooth jazz became successful, garnering significant radio airplay. Other styles and genres abound in the 2000s, such as Latin and Afro-Cuban jazz."
})
headers = {
  'Content-Type': 'application/json'
}


# POST the Request
response = requests.request("POST", url, headers=headers, data=payload)
print(response.text)

