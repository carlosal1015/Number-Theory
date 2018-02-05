\NeedsTeXFormat{LaTeX2e}[2017/01/01] % More information: https://www.latex-project.org/news/latex2e-news/
\ProvidesClass{gemnumber}[16/01/2018 V2.0 clase de documento del Grupo Estudiantil de Matemática de la Facultad de Ciencias para el curso de Teoría de números]
\LoadClass[oneside,a5paper]{memoir} % Clase heredado de memoir. % Se puede desarrollar chapter dentro de article.
\RequirePackage[utf8]{inputenc} % Not utf8x.
\RequirePackage[T1]{fontenc}
\RequirePackage{libertine}
\RequirePackage[spanish]{babel}
\spanishdatedel
\RequirePackage[pdftex]{graphicx}
\RequirePackage[usenames,dvipsnames,x11names,table,svgnames]{xcolor}
\RequirePackage[colorlinks=true,urlcolor=blue,linkcolor=black,anchorcolor=black,citecolor=black]{hyperref}
\RequirePackage{mathtools,amssymb,amsfonts,amsmath,amsthm,mathrsfs,bm,times,mathabx}
\RequirePackage[makeroom]{cancel}
\RequirePackage{minted}
\RequirePackage{multicol}
\RequirePackage{background}
\RequirePackage{float}
\RequirePackage{caption,subcaption}
\RequirePackage{enumitem}
\RequirePackage[top = 1cm, bottom = 1cm, left = 1cm, right = 1cm, headsep = 2mm, headheight = 0.5cm, foot = 5mm, marginparwidth = 2cm, marginparsep = 5mm]{geometry}

\hypersetup{pdfinfo={
		Title={Apuntes de clases de Teoría de números},
		Author={Jimmy Espinoza Palacios},
		Keywords={Divisibilidad, Congruencias, Funciones aritméticas},
		Subject={Teoría de números},
		Producer={TeXstudio 2.12.6},
		Creator={pdfTeX Version 3.14159265-2.6-1.40.18 TeX Live 2018 Debian}
}}
\backgroundsetup{%
	scale=1,       %% change accordingly
	angle=0,       %% change accordingly
	opacity=.1,    %% change accordingly
	contents={\begin{tikzpicture}[remember picture,overlay]
		\node at ([yshift=11pt,xshift=5pt]current page.center) {\includegraphics[width=8.1cm]{gemtransparente}};    %% yshift and xshift for example only
		\end{tikzpicture}}
}
\ProcessOptions \relax % Para que se procesen heredados y se busque desde la clase memoir.