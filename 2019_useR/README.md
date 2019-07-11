The following needs to be run to convert the HTML to PDF.
Docker is required and needs to be configured correctly.

```r
xaringan::decktape("2019-useR/xaringan/user2019_mlr3.html",
                   "2019-useR/mlr3-useR-2019.pdf", docker = TRUE)
```
