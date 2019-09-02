#!/bin/bash 
cd tex

main_web_file="../webs/packagehandoff-main.web"
main_tex_file="packagehandoff-main.tex"



if [ $# -eq 0 ]; then 
        # no arguments passed, run in default 
        # mode, by weaving and tangling  
        nuweb -r -v $main_web_file  
        pdflatex -interaction=nonstopmode -halt-on-error  -shell-escape $main_tex_file   

	asy ./asytmp/*.asy -o ./asytmp/
	
        bibtex $main_tex_file  
        pdflatex -interaction=nonstopmode -halt-on-error  -shell-escape $main_tex_file 
        nuweb -r -v $main_web_file 
        pdflatex -interaction=nonstopmode -halt-on-error  -shell-escape $main_tex_file 
 	mv  packagehandoff-main.pdf  ../

		
elif [ $1=="--tangle-only" ] ; then 
        # Only extract the source code 
        nuweb -t -v $main_web_file 
else   
        echo "Option not recognized."
fi 
