%
% LaTeX class for GPUSPH documentation
%

\ProvidesClass{GPUSPHtemplate}[2016/07/07]

%----------------------------------------------------------------------------------------
%	DECLARE CLASS AND GEOMETRY
%----------------------------------------------------------------------------------------

\LoadClass[a4paper,twoside,12pt]{article}

\RequirePackage[letterpaper]{geometry}

%----------------------------------------------------------------------------------------
%	VARIOUS REQUIRED PACKAGES
%----------------------------------------------------------------------------------------

\RequirePackage{lmodern}

% UTF-8 input encoding
\RequirePackage[utf8]{inputenc}
% T1 font encoding
\RequirePackage[T1]{fontenc}

% Allow for underscores in the text (without using \_)
\AtBeginDocument{%
  \begingroup\lccode`~=`_%
  \lowercase{\endgroup\let~}_%
  \catcode`_=12
}

% URL management
\RequirePackage{url}
\RequirePackage[hidelinks]{hyperref}

% TODO notes
\RequirePackage{todonotes}

\RequirePackage{nth}

% listings
\RequirePackage{listings}
\lstloadlanguages{sh,make,C++}
\lstset{
 basicstyle=\ttfamily,
 xleftmargin=2\parindent,
 xrightmargin=2\parindent,
}

\lstnewenvironment{shellcode}[1][]{\lstset{language=sh,#1}}{}
\lstnewenvironment{ccode}[1][]{\lstset{language=C++,#1}}{}


% graphics inclusion
\RequirePackage{graphicx}
\RequirePackage{caption}
\RequirePackage{subcaption}
\RequirePackage{float}

% extra mathematical symbols, full AMS math support
\RequirePackage{amssymb,amsmath,bm}

% wrap text around figures
\RequirePackage{wrapfig}

% bibliography
\RequirePackage[round]{natbib}
\bibliographystyle{plainnat}

% indentation
\newlength\tindent
\setlength{\tindent}{\parindent}
\setlength{\parindent}{0pt}

% Appendix
\RequirePackage[titletoc]{appendix}

% common math shortcuts
\newcommand{\be}{\begin{equation}}
\newcommand{\en}{\end{equation}}
\newcommand{\bx}{\mathbf{x}}
\newcommand{\uvec}[1]{\underline{#1}}
\renewcommand{\vec}[1]{\bm{#1}}
\newcommand{\td}{\text{d}}
\newcommand{\tdv}[2]{\frac{\td #1}{\td #2}}
\newcommand{\tddv}[2]{\frac{\td^2 #1}{\td #2^2}}
\newcommand{\pdv}[2]{\frac{\partial #1}{\partial #2}}
\newcommand{\pddv}[2]{\frac{\partial^2 #1}{\partial #2 ^2}}
\newcommand{\abs}[1]{\ensuremath{\left|#1\right|}}
\newcommand{\lap}{\nabla^2}
\newcommand{\ie}{\textit{i.e.}~}
\newcommand{\eg}{\textit{e.g.}~}
\newcommand{\etal}{\textit{et al.}~}
\newcommand{\sumF}{\underset{b \in \mathcal{F}}{\sum}}
\newcommand{\sumP}{\underset{b \in \mathcal{P}}{\sum}}
\newcommand{\sumS}{\underset{s \in \mathcal{S}}{\sum}}
\newcommand{\Grad}{\textbf{G}}
\newcommand{\Div}{D}
\newcommand{\Lap}{\textbf{L}}

% current version
\newcommand{\version}{5.0}
\newcommand{\currentver}{version~\version}

% text macros
\newcommand{\nvidia}{\textsc{nvidia}}
\newcommand{\cpp}{{\sffamily C\ttfamily++}}
\newcommand{\cu}{{\sffamily cu}}

% text emphasis
\newcommand{\cmd}[1]{\lstinline[language=sh]{#1}}
