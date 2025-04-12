import combnet

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import penn
else:
    from combnet import penn

penn.model.crepe.Crepe()