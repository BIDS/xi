
typedef struct {
    double pos[3];    
    double vel[3];
    double weight;
} particle;

typedef struct {
  int bintype;
  int logxopt;
  int nx;
  double minx;
  double dx;
  int logyopt;
  int ny;
  double miny;
  double dy;
  int realorzspace;
  int periodicopt;
  int nbins2d;
  //wp only:
  double rpimax;
  int zspaceaxis;
} xibindat;

int paircount(dataset * p1, dataset * p2, xibindat b);
