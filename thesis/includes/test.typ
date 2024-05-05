#let title = [
  Attention is all you need
]
#set page(
  paper: "us-letter",
  header: align(right + horizon)[
    #title
  ],
  numbering: "1",
)
#set par(justify: true)
#set text(
  font: "Linux Libertine",
  size: 11pt
)


#align(center, text(17pt)[
  *#title*
])

#grid(
  columns: (1fr),
  align(center)[
    Rui Xu \
    PALM of SEU \
    #link("andrewrey.cc@gmail.com")
  ]
)

#align(center)[
  #set par(justify: false)
  *Abstract* \
  #lorem(100)
]

#show heading: it => [
  #set align(center)
  #set text(12pt, weight: "regular")
  #block(smallcaps(it.body))
]

#show: rest => columns(2, rest)

= Introduction
#lorem(300)

= Related Work
#lorem(500)