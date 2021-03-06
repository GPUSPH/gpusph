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

% Granular rheology shortcuts
\newcommand{\tens}[1]{\vec{\vec{#1}}\,}
\newcommand{\rhoeq}{\rho_{\text{eq}}}
\newcommand{\etaeff}{\eta_{\text{eff}}}
\newcommand{\peff}{p_{\text{eff}}}
\newcommand{\rhow}{\rho_{\text{w}}}
\newcommand{\rhog}{\rho_{\text{g}}}
\newcommand{\tauy}{\tau_{\text{y}}}
\newcommand{\ptot}{p_{\text{tot}}}
\newcommand{\ppw}{p_{\text{pw}}}
\newcommand{\wab}{w_{ab}}
\newcommand{\gwab}{\vec{\nabla} w_{ab}}
\newcommand{\gGamAS}{\vec{\nabla} \gamma_{as}}
\newcommand{\gGamA}{\vec{\nabla} \gamma_{a}}
\newcommand{\ns}{\vec{n}_{\,s}}
\newcommand{\eab}{\vec{e}_{\,ab}}
\newcommand{\sumb}{\sum\limits_{b\in \left(\mathcal{F} \cup \mathcal{V} \right)}}
\newcommand{\sumbb}{\sum\limits_{b\in{\mathcal{F}}}}
\newcommand{\sumbfv}{\sum\limits_{b\in \mathcal{M} \cup \mathcal{V}}}
\newcommand{\sumbf}{\sum\limits_{b\in \mathcal{M}}}
\newcommand{\sumbv}{\sum\limits_{b\in \mathcal{V}}}
\newcommand{\sums}{\sum\limits_{s\in \mathcal{S}}}
\newcommand{\sumbi}{\sum\limits_{b\in \mathcal{M} \cap \mathcal{I}}}
\newcommand{\sumbfnoi}{\sum\limits_{b\in \mathcal{M} \setminus \mathcal{I}}}
\newcommand{\dg}{d_{\text{g}}}
\newcommand{\rab}{r_{ab}}
\newcommand{\vnabla}{\vec{\nabla}}
\newcommand{\vu}{\vec{u}} %velocity vector



% current version
\newcommand{\version}{5.0}
\newcommand{\currentver}{version~\version}
\newcommand{\reldate}{June 2019}

% text macros
\newcommand{\nvidia}{\textsc{nvidia}}
\newcommand{\cpp}{{\sffamily C\ttfamily++}}
\newcommand{\cu}{{\sffamily cu}}

% text emphasis
\newcommand{\cmd}[1]{\lstinline[language=sh]{#1}}
