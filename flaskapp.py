import subprocess
import uuid
from flask import Flask
from flask import request
from flask import render_template
from flask import redirect
from flask import url_for
from flask import session


app = Flask(__name__)
app.secret_key = "\x08\x03t\x198\n\xa7\x95\xa1!#\x90&6\xb6G|\x04\xbbF\xccN\xe0\x00"

@app.route('/', methods=['GET', 'POST'])
def submit_input():
  error = None
  if request.method == 'POST':
    while True:
      try:
        rawInputPath = "/work/web_interface/tmp/" + str(uuid.uuid4()) + ".txt"
        with open(rawInputPath, "w") as outputFile:
          outputFile.write(request.form["rawtext"])
        session['papers'] = recommend_papers(rawInputPath)
        break
      except ValueError:
        print "Waiting for Spark to get free..."
    return redirect(url_for("show_recommendations"))

  return render_template('home.html', error=error)

@app.route('/recommendations')
def show_recommendations():
  return render_template('recommendations.html')

def recommend_papers(rawInputPath):
  cmd = "spark-submit /work/web_interface/recommend_papers.py " + rawInputPath
  cmd = cmd.split(' ')
  output = subprocess.Popen(cmd, stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()
  return output[0].split('\n')[-2].split(' ')

if __name__ == '__main__':
  app.run()
  rawInput = """We analyze the effective triplet interactions between the centers of star polymers in a good solvent. 
  Using an analytical short distance expansion inspired by scaling theory, we deduce that the triplet part of the three-star 
  force is attractive but only 11% of the pairwise part even for a close approach of three star polymers. We have also performed 
  extensive computer simulations for different arm numbers to extract the effective triplet force. The simulation data show good 
  correspondence with the theoretical predictions. Our results justify the effective pair potential picture even beyond the star 
  polymer overlap concentration."""
  # recommend_papers(rawInput)
