import requests
import os


def main():
    bases = ('C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B')
    types = ('pp', 'mf', 'ff')
    address = r'http://theremin.music.uiowa.edu/sound%20files/MIS/Piano_Other/piano/'
    for typ in types:
        for k in range(88):
            note = bases[(k - 3) % 12] + str(((k - 3) // 12) + 1)
            filename = f'Piano.{typ}.{note}.aiff'
            print(f'Downloading {filename}')
            r = requests.get(address + filename)
            with open(os.path.join('samples', filename), 'wb') as f:
                f.write(r.content)


if __name__ == '__main__':
    main()
