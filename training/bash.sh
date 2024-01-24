for i in $(ls data/csv/cleaned/); do
    sort -R "data/csv/cleaned/$i" | head -n 50000 >> data/europarl.csv
    echo "finished loading 50000 random rows of $i into europarl.csv"
done
