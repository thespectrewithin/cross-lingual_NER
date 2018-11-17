import argparse

parser = argparse.ArgumentParser(description='transfer training data')

parser.add_argument('--word_translation', type=str, help='word-to-word translation file')
parser.add_argument('--training_data', type=str, help='source training data file')
parser.add_argument('--same_script', action='store_true', help='if the two languages use the same script')
parser.add_argument('--output', type=str, help='output transferred training data file')

args = parser.parse_args()

word_dict = {}
for line in open(args.word_translation):
    l = line.strip().split()
    word_dict[l[0]] = l[1].lower()

output = open(args.output, 'w')

for line in open(args.training_data, 'r'):
    if len(line.strip()) > 0:
        l = line.strip().split()
        word = l[0]
        if args.same_script:
            if l[1][2:] == 'PER':
                output.write(word + ' ' + l[1] + '\n')
            else:
                if word.lower() in word_dict:
                    temp = word_dict[word.lower()]
                    if word.isupper():
                        word = temp.upper()
                    elif word[0].isupper():
                        word = temp[0].upper() + temp[1:]
                    else:
                        word = temp
                output.write(word + ' ' + l[1] + '\n')
        else:
            if word.lower() in word_dict:
                word = word_dict[word.lower()]
            output.write(word + ' ' + l[1] + '\n')
    else:
        output.write(line)

output.close()
