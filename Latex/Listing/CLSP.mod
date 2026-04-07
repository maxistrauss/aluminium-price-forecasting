// Parameter, Teil 1:
int T = ...;      // Planungszeitraum.
int K = ...;      // Produkte.
int J = ...;      // Ressourcen.
int M = ...;      // Große Zahl.   

// Wertebereiche:
range Produkt = 1..K;
range Ressource = 1..J;
range Planungszeitraum = 1..T;
range PlanungszeitraumNull = 0..T;

// Variablen:
dvar int+ q[Produkt][Planungszeitraum];         // Losgrössen. 
dvar int+ y[Produkt][PlanungszeitraumNull];     // Lagerbestände.
dvar boolean gamma[Produkt][Planungszeitraum];  // Rüstvariablen. 

// Parameter, Teil 2:
int b[Ressource][Planungszeitraum] = ...;   // Kapazitäten. 
int	d[Produkt][Planungszeitraum] = ...;     // Nettobedarfe.
float h[Produkt] = ...;                     // Lagerkostensätze.
float s[Produkt] = ...;                     // Rüstkostensätze.
// Stückbearbeitungszeiten:
int tb[Produkt][Ressource] = ...;    
int tr[Produkt][Ressource] = ...;           // Rüstzeiten. 
int z[Produkt] = ...;                    // Mindestvorlaufzeiten.
int y0[Produkt] = ...;                   // Anfangslagerbestände.

// Minimierung der Gesamtkosten
minimize                          
   sum (k in Produkt, t in Planungszeitraum) 
		(s[k] * gamma[k][t] + h[k] * y[k][t]);
   
constraints {
	// Lagerbilanzgleichungen:
	forall(k in Produkt){					  
		forall(t in 1..(z[k])){
			y[k][t-1] - d[k][t] == y[k][t];
		}
		forall(t in (z[k]+1)..T){
			y[k][t-1] + q[k][t-z[k]] - d[k][t] == y[k][t];
		}
	}
	// Kapazitätsbedingungen:
	forall(j in Ressource, t in Planungszeitraum){	
		sum(k in Produkt)(tb[k][j] * q[k][t] + tr[k][j] * gamma[k][t])
		                  <= b[j][t];
	}
	// Rüstbedingungen:
	forall(k in Produkt, t in Planungszeitraum){	
		q[k][t] - M * gamma[k][t] <= 0;
	}
	// Lageranfangsbestände:
	forall(k in Produkt){							
		y[k][0] == y0[k];	
	}
};