import argparse

from nltk.tokenize import sent_tokenize, word_tokenize
import collections
import ast
import json

'''
Build the tokenized document.
input: "Young rebels.\n We are facing a major threat. Children have been (...)." becomes:
output: "<d> <p> <s> </p> <p> <s> We are facing a major threat . </s> <s> Children have been ( ... ) . </s> </p> </d>'
'''
def _pre_process(reader, writer):
  # Each record has the format { "id": "fc7...", "title": "Veterans ...", "content: "Bla bla bla" }
#  i = 0
#  while i < 10:
#    i += 1
  while True:
    record = reader.readline()
    if len(record) == 0:
      break
    dict = ast.literal_eval(record)
    out = { }
    for key,value in dict.items():
      if key == 'title' or key == 'content':
        out[key] = _build_document(value)
    writer.write(json.dumps(out) + '\n')

def _build_document(text):
  paragraphs = text.split('\n') # Break paragraphs
  out = ''
  for paragraph in paragraphs:
    if len(paragraph) > 2:
      sentences = sent_tokenize(paragraph)
      i = 0
      for sentence in sentences:
        i += 1
        # Batchreader is limiting to two sentences, so we stay up to two
        if (i > 2):
          break;
        words = word_tokenize(sentence)
        out = out + '<p> <s> ' + ' '.join([ word for word in words ]) + ' </s> </p> '
  return ('<d> ' + out.strip() + ' </d>')

def _build_vocab(reader, writer):
  counter = collections.Counter()
  while True:
    record = reader.readline()
    if len(record) == 0:
      break;
    dict = json.loads(record)
    for k,v in dict.items():
      counter.update(v.split())
  for word,count in counter.most_common(25000):
    writer.write(word + ' ' + str(count) + '\n')
  writer.write('<UNK> 0\n')
  writer.write('<PAD> 0\n')

def main(args):
  if args.command == 'pre_process':
    _pre_process(args.in_file, args.out_file)
  elif args.command == 'build_vocab':
    _build_vocab(args.in_file, args.out_file)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--in_file', type=argparse.FileType(mode='r',encoding='utf-8'), required=True)
parser.add_argument('--out_file', type=argparse.FileType(mode='w', encoding='utf-8'), required=True)
parser.add_argument('--command', required=True)

args = parser.parse_args()

if __name__ == "__main__":
  main(args)
