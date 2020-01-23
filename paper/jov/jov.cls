\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{jov}[2008/12/16 Journal of Vision class]

\renewcommand\normalsize{\fontsize{10pt}{12pt}\selectfont}

\setlength{\textwidth}{6.5in}
\setlength{\textheight}{8in}


\LoadClass{article}
\RequirePackage{color}
\RequirePackage{times}
\RequirePackage{multirow}
\RequirePackage[colorlinks=true, linkcolor=blue, urlcolor=blue, citecolor=blue, anchorcolor=blue, pageanchor=true]{hyperref}
\RequirePackage{jovcite}
\RequirePackage{pifont}
%\RequirePackage[alpine, misc]{ifsym}

%%% Code from patchcmd.sty:
% (Copyright 2000 Michael John Downes)
\newcommand{\patchcommand}[1]{%
  \expandafter\patchcmd@a\meaning#1??->@\@nil#1%
}
\long\def\patchcmd@a#1#2#3->#4#5\@nil#6{%
%%  \wrs{\string#6: [#1] [#2] [#3]->[#4]}%
  \ifx @#4\relax \patchcmdError#6#1%
    \expandafter\@gobbletwo % discard the other two arguments
  \else
    \if l#2\toks@{\patchcmd@e{}#6}% l in this position means \long
    \else \toks@{\patchcmd@e*#6}%   not \long
    \fi
    \patchcmd@b #3@#4#5 ? ? ? \@nil#6%
    \expandafter\the\expandafter\toks@
  \fi
}
\def\patchcmd@b#1:#2@#3#4 #5#6 #7 #8\@nil#9{%
%%  \wrs{[#1] [#2] [#3] [#4] [#5] [#6] ARG7=[#7] [#8]}%
  \if \ifx @#7@\expandafter
      \ifx\csname #6\endcsname#9T\else F\fi\else F\fi T%
    \toks@\expandafter{\expandafter\patchcommand\csname #6 \endcsname}%
  \else
    \ifx @#2@% No arguments
      \toks@\expandafter{\the\toks@ 0}%
    \else
      \patchcmd@c 0#2{\string##}0%
    \fi
  \fi
}
\def\patchcmd@c#1#2#3{%
  \if\string###2%      % yes it's a # token
    \ifodd 0#31 % and it's followed by a number
      \if 0#3\patchcmd@d#1\fi % number=0? then we're done
    \else \patchcmd@d D% # not a number: must be a delimited arg
    \fi
  \else \patchcmd@d D% not a # token: must be a delmited arg
  \fi
  \patchcmd@c#3%
}
\def\patchcmd@d#1{%
  \if D#1%
    \PackageError{patchcmd}{Cannot change a macro that has
      delimited arguments}\@ehd
  \else
    \toks@\expandafter{\the\toks@ #1}%
  \fi
  \begingroup
  \aftergroup\@gobble
  \let\patchcmd@c\endgroup
}
\def\patchcmd@e#1#2#3#4#5{%
  \begingroup
  \edef\@##1{%
    \@temptokena\noexpand\expandafter{%
      \noexpand#2%
        \ifnum#3>0 {####1}\ifnum#3>1 {####2}\ifnum#3>2 {####3}%
        \ifnum#3>3 {####4}\ifnum#3>4 {####5}\ifnum#3>5 {####6}%
        \ifnum#3>6 {####7}\ifnum#3>7 {####8}\ifnum#3>8 {####9}%
        \fi\fi\fi\fi\fi\fi\fi\fi\fi
      ##1%
    }%
  }
  \@{#5}%
  \edef\@##1{\endgroup
    \noexpand\renewcommand#1\noexpand#2\ifcase#3 \else [#3]\fi
    {##1\the\@temptokena}}%
  \@{#4}%
%%  \show#2%
}

%%% End of patchcmd code

\textwidth 190mm
\hoffset -35mm
\textheight 230mm
\voffset -20mm	
\linespread{1.3}

\setlength{\parindent}{7mm}

%%% Title:
\definecolor{lightblue}{rgb}{0.02, 0.55, 0.86}
\newcommand{\blueframe}[1]{\color{white}\colorbox{lightblue}{\makebox[0.98\linewidth][l]{#1}}}
\renewcommand{\section}{%
	\@startsection
		{section}{1}{0pt}{2ex}%
		{2ex}{\Large\bf\sffamily\noindent\blueframe}%
}
\renewcommand{\subsection}{%
	\@startsection
		{section}{1}{0pt}{2ex}%
		{1ex}{\large\bf\sffamily\noindent}%
}
\renewcommand{\subsubsection}{%
	\@startsection
		{section}{1}{0pt}{2ex}%
		{1ex}{\sffamily\slshape\bfseries\noindent}%
}
\setcounter{secnumdepth}{0}	% no section numbering

\newcommand{\@titlestring}{}
\newcommand{\@abstractstring}{}
\newcommand{\@authorsstring}{}
\newcommand{\@keywordstring}{}
\newcommand{\@authorsheading}{}
\newcommand{\@authorsheadingtmp}{}
%\newcommand{\@lastauthor}{}
%\newcommand{\@twoauthorsheading}{}	% special heading for 2 authors
\renewcommand{\title}[1]{\renewcommand\@titlestring{#1}}
\renewcommand{\abstract}[1]{\renewcommand\@abstractstring{#1}}
\newcommand{\keywords}[1]{\renewcommand\@keywordstring{#1}}
\newcounter{nauthors}
% author arguments: Family name, First name(s), Institution, Address, homepage, email
\renewcommand{\author}[6]{%
	\stepcounter{nauthors}%
	\ifnum\value{nauthors}=1%
		\renewcommand\@authorsheading{#1}
		\renewcommand\@authorsheadingtmp{#1}
	\else%
		\edef\previous@uthors{\@authorsheadingtmp}% expanded definition
		\patchcommand\@authorsheadingtmp{}{, #1}%
		%\renewcommand\@lastauthor{#1}%
		\ifnum\value{nauthors}=2%
			\edef\@authorsheading{\previous@uthors{} \& #1}%
		\else%
			\edef\@authorsheading{\previous@uthors{}, \& #1}%
		\fi%
	\fi%
	\patchcommand\@authorsstring{}{%
		\multirow{2}{*}{\Large\bf\textsf{#2 #1}} & \sf #3 & \multirow{2}{*}{\Huge\href{#5}{\ding{118}} \href{mailto:#6}{\ding{41}}}\\%
			 & \sf #4 & \\%
			 \multicolumn{3}{c}{\hspace*{0.97\textwidth}}\\}%
}

\renewcommand{\maketitle}{%
	\thispagestyle{myheadings}%
	{{\noindent\huge\sffamily\fontseries{b}\selectfont \@titlestring}\vspace*{10mm}\par%
	\noindent\begin{tabular}{lrr}%
	\@authorsstring%
	\end{tabular}\vspace*{5mm}\par%
	\noindent\sffamily\@abstractstring\vspace*{5mm}\par%
	Keywords: \@keywordstring\vspace*{10mm}\par}%
}

%%% header for title page:
\def\ps@myheadings{\let\@mkboth\markboth
	\def\@oddfoot{\protect{\hbox to\textwidth{\footnotesize\sf doi:\hspace*{7cm} Received: \today\hfill ISSN 1534--7362 \textcopyright{} 20?? ARVO}}}
	\def\@oddhead{\protect{\hbox to\textwidth{\footnotesize\sf \textsl{Journal of Vision} (20??) ?, 1--?\hspace*{4cm} \href{http://journalofvision.org/}{http://journalofvision.org/?/?/?}\hfill\thepage}}}
	\def\sectionmark##1{\markright{\ifnum \c@secnumdepth >\z@
		\thesection. \ \fi ##1}}}

%%% default header:
\def\ps@plain{\let\@mkboth\markboth
	\def\@oddfoot{}
	\def\@oddhead{%
	\protect{\footnotesize\sf \textsl{Journal of Vision} (20??) ?, 1--?\hfill\@authorsheading\hfill\hspace*{37mm}\thepage}}
	%\ifnum\value{nauthors}=2%
	%{\protect{\footnotesize\sf \textsl{Journal of Vision} (20??) ?, 1--?\hfill\@twoauthorsheading\hfill\hspace*{37mm}\thepage}}%
	%\else%
	%{\protect{\footnotesize\sf \textsl{Journal of Vision} (20??) ?, 1--?\hfill\previous@uthors, \& %\@lastauthor\hfill\hspace*{37mm}\thepage}}%
	%\fi}%
	\def\sectionmark##1{\markright{\ifnum \c@secnumdepth >\z@
		\thesection. \ \fi ##1}}}

\pagestyle{plain}

% font in captions
\newcommand{\captionfonts}{\footnotesize\sf}

\long\def\@makecaption#1#2{%
 \vskip\abovecaptionskip
 \sbox\@tempboxa{{\captionfonts #1: #2}}%
 \ifdim \wd\@tempboxa >\hsize
 {\captionfonts #1: #2\par}
 \else
 \hbox to\hsize{\hfil\box\@tempboxa\hfil}%
 \fi
 \vskip\belowcaptionskip}

