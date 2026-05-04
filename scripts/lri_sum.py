"""
Likelihood Ratio com contagem de células ardidas.

Este módulo não altera o código fonte do GLASS.
Cria uma variante de HeuristicLri em que:
- o LR é calculado com 'count', tal como no GLASS original;
- o aspect é opcional;
- para LULC, cada período usa o total ardido do seu próprio período.
"""

import os

from glass.ete.lri import HeuristicLri


class HeuristicLriCount(HeuristicLri):
    """
    Variante do HeuristicLri que calcula o LR com contagem de células ardidas.

    Diferença face ao GLASS original:
    - permite calcular o LRI com ou sem aspect;
    - para LULC, calcula o total ardido separadamente para cada período temporal.
    """

    def get_lri(self, events, rstvar, lri, burncells, tcells):
        """
        Calcula o LRI para uma variável específica.
        """

        from glass.rst.zon.grs import rstatszonal
        from glass.rst.alg import grsrstcalc

        if burncells == 0:
            raise ValueError(
                f"O raster de eventos {events} não tem ocorrências válidas."
            )

        # Contagem de células ardidas por classe da variável.
        burn_cells_cls = rstatszonal(
            rstvar,
            events,
            "count",
            f"burn_{rstvar}"
        )

        # Número de células existentes em cada classe da variável.
        cells_cls = rstatszonal(
            rstvar,
            rstvar,
            "count",
            f"class_{rstvar}"
        )

        _lri = grsrstcalc(
            (
                f"(double({burn_cells_cls}) / double({cells_cls}))"
                f" / "
                f"({float(burncells)} / {float(tcells)})"
            ),
            lri
        )

        return _lri

    def calc_lri(self, out=None):
        """
        Calcula o LRI final usando contagem de células ardidas.
        """

        from glass.prop.rst import count_cells
        from glass.rst.alg import grsrstcalc
        from glass.it.rst import grs_to_rst

        if not self.vardem:
            raise ValueError("DEM var is missing")

        if not self.varslope:
            raise ValueError("SLOPE var is missing")

        if not self.rst_lulc:
            raise ValueError("LULC var is missing")

        if not self.gevents:
            raise ValueError("Events raster is missing")

        # Evita manter resultados de uma execução anterior no mesmo objecto.
        self.lri_results = {}

        # Número total de células válidas da área de estudo.
        filevardem = grs_to_rst(
            self.vardem,
            os.path.join(self.ws, self.loc, f"{self.vardem}.tif"),
            dtype="Int32"
        )

        ncells = count_cells(filevardem)

        # Contagem total de células ardidas no raster global.
        bcells_global = count_cells(
            grs_to_rst(
                self.gevents,
                os.path.join(self.ws, self.loc, f"{self.gevents}.tif"),
                dtype="Int32"
            )
        )

        _vars = {
            "dem": self.vardem,
            "slope": self.varslope,
            "lulc": self.rst_lulc
        }

        # O aspect só entra se tiver sido importado no notebook.
        varaspect = getattr(self, "varaspect", None)

        if varaspect:
            _vars["aspect"] = varaspect

        for k in _vars:
            if type(_vars[k]) == dict:
                exp, y = [], 0

                for lulc in _vars[k]:
                    revents = _vars[k][lulc]["burned"]

                    # Cada LULC usa o total ardido do seu próprio período.
                    bcells_lulc = count_cells(
                        grs_to_rst(
                            revents,
                            os.path.join(self.ws, self.loc, f"{revents}.tif"),
                            dtype="Int32"
                        )
                    )

                    _lri = self.get_lri(
                        revents,
                        lulc,
                        f"lri_{lulc}",
                        bcells_lulc,
                        ncells
                    )

                    weight = _vars[k][lulc]["weight"]
                    y += weight

                    exp.append(f"({_lri} * {str(weight)})")

                self.lri_results[k] = grsrstcalc(
                    f"({' + '.join(exp)}) / {str(y)}",
                    f"lri_{k}"
                )

            else:
                self.lri_results[k] = self.get_lri(
                    self.gevents,
                    _vars[k],
                    f"lri_{k}",
                    bcells_global,
                    ncells
                )

        self.final_lri = grsrstcalc(
            f" + ".join(list(self.lri_results.values())),
            "final_lri"
        )

        if out:
            grs_to_rst(self.final_lri, out, dtype="Float64")

        return out