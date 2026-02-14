"""
Data Providers for Cleanlab Tasks.

This module provides concrete implementations of data providers for various datasets.
These can be customized or replaced to use different datasets with the same task logic.
"""

from cleanlab_demo.data.providers.anomaly import (
    CaliforniaHousingAnomalyProvider,
    ForestCoverAnomalyProvider,
    KDDCup99Provider,
    SyntheticAnomalyProvider,
)
from cleanlab_demo.data.providers.multiannotator import (
    MovieLens100KProvider,
)
from cleanlab_demo.data.providers.multiclass import (
    CovtypeDataProvider,
    SKLearnDatasetProvider,
)
from cleanlab_demo.data.providers.multilabel import (
    EmotionsDataProvider,
    OpenMLMultilabelProvider,
)
from cleanlab_demo.data.providers.outlier import (
    CaliforniaHousingOutlierProvider,
)
from cleanlab_demo.data.providers.regression import (
    BikeSharingDataProvider,
    CaliforniaHousingDataProvider,
    TabularRegressionProvider,
)
from cleanlab_demo.data.providers.token import (
    ConlluDataProvider,
    UDEnglishEWTProvider,
)
from cleanlab_demo.data.providers.vision import (
    PennFudanPedProvider,
)


__all__ = [
    "BikeSharingDataProvider",
    "CaliforniaHousingAnomalyProvider",
    "CaliforniaHousingDataProvider",
    "CaliforniaHousingOutlierProvider",
    "ConlluDataProvider",
    "CovtypeDataProvider",
    "EmotionsDataProvider",
    "ForestCoverAnomalyProvider",
    "KDDCup99Provider",
    "MovieLens100KProvider",
    "OpenMLMultilabelProvider",
    "PennFudanPedProvider",
    "SKLearnDatasetProvider",
    "SyntheticAnomalyProvider",
    "TabularRegressionProvider",
    "UDEnglishEWTProvider",
]
