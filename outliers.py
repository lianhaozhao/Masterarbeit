
outliers = [
        # no signal outliers
        'DC_T194_R05_I301',
        'DC_T194_R05_I327',
        'DC_T194_R05_I331',
        
        'DC_T194_R50_I289',
        'DC_T194_R50_I297',
        'DC_T194_R50_I303',
        'DC_T194_R50_I312',
        
        'DC_T197_R10_I343',
        
        'DC_T197_R15_I390',
        'DC_T197_R15_I400',
        
        'DC_T197_R50_I188',
        'DC_T197_R50_I403',
        
        # scale problem outlier
        'DC_T197_R15_I88',
        'DC_T194_R50_I325'
    ]

# trigger signal shift
outliers.extend([f'HC_T194_R05_I{i}' for i in range(349, 501)])

# outliers in test set
outliers.extend([
                'DC_T185_R05_I367',
                'DC_T185_R05_I388',
                'DC_T185_R05_I391',
                'DC_T185_R05_I396',
                'DC_T185_R05_I399',
                'DC_T185_R05_I415',
                'DC_T185_R05_I489',
                'DC_T185_R05_I493',
                'DC_T185_R05_I496',
                 
                'DC_T185_R10_I230',
                 ])

outliers.extend([
                'DC_T188_R05_I122',
                'DC_T188_R05_I132',
                'DC_T188_R05_I143',
                'DC_T188_R05_I206',
                'DC_T188_R05_I218',
                'DC_T188_R05_I222',
                'DC_T188_R05_I227',
                 
                'DC_T188_R15_I0',
                'DC_T188_R15_I1',
                 ])

outliers.extend([
                'DC_T191_R05_I399',
                'DC_T191_R05_I402',
                'DC_T191_R05_I407',
                'DC_T191_R05_I413',
                'DC_T191_R05_I421',
                'DC_T191_R05_I430',
                 
                'DC_T191_R15_I287',
                ])

outliers.extend([
                'HC_T188_R05_I0',
                'HC_T188_R05_I200',
                'HC_T188_R05_I437',
                'HC_T188_R05_I426',
                
                'HC_T188_R10_I183',
                'HC_T188_R10_I405',
                'HC_T188_R10_I25',
                'HC_T188_R10_I329',
                'HC_T188_R10_I29',
                'HC_T188_R10_I50',
                'HC_T188_R10_I394',
                'HC_T188_R10_I306',
                'HC_T188_R10_I414',
                'HC_T188_R10_I29',
                'HC_T188_R10_I57',
                ])

outliers.extend([
    'HC_T185_R45_I474'
])

outliers.extend([f'HC_T185_R40_I{i}' for i in range(459, 468)])
outliers.extend([f'HC_T185_R40_I{i}' for i in range(486, 501)])

outliers.extend([
    'HC_T185_R15_I369',
    'HC_T185_R15_I40',
    'HC_T185_R15_I99',
])

outliers.extend([
    'HC_T185_R20_I0'
])

outliers.extend([f'HC_T185_R20_I{i}' for i in range(150, 160)])
outliers.extend([f'HC_T185_R20_I{i}' for i in range(497, 501)])

outliers.extend([f'HC_T185_R30_I{i}' for i in range(85, 100)])
outliers.extend([f'HC_T185_R30_I{i}' for i in range(499, 501)])


outliers.extend([
    'HC_T185_R10_I7',
    'HC_T185_R10_I14'
])
outliers.extend([f'HC_T185_R10_I{i}' for i in range(213, 216)])
outliers.extend([f'HC_T185_R10_I{i}' for i in range(486, 501)])