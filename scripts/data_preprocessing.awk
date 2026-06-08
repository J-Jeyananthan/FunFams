#check if 1-1 mapping
| awk -F'\t' '($1 != $2){print "Mismatch at line", NR, "FASTA=" $1, "MAP=" $2; exit 1} END{print "OK:", NR, "records matched in order"}'


#check counts of funfams before preprocessing
awk -F'\t' '{c[$2]++} END{for (f in c) print f "\t" c[f]}' \
>   funfams-4.3-c123-mapping.txt > funfam_counts.txt


#remove duplicate IDs from mapping list
awk -F'\t' '
>   NR==FNR { bad[$2]=1; next }   # first file: collect IDs to remove
>   !($1 in bad)                  # second file: keep only IDs NOT in bad[]
> ' duplicate_raw_headers.txt funfams-4.3-c123-mapping.txt \
> > duplicates_removed-funfams-4.3-c123-mapping.txt


#remove ids from funfams with <3 members
awk -F'\t' '
>   NR==FNR { c[$2]++; next }
>   c[$2] < 3 { print > "removed_small_funfams.txt"; next }
>   { print }
> ' duplicates_removed-funfams-4.3-c123-mapping.txt \
>   duplicates_removed-funfams-4.3-c123-mapping.txt \
> > min3-funfams-4.3-c123-mapping.txt
